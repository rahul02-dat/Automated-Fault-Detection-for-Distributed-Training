import torch
import torch.nn as nn
import math
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from src.workloads.common import Workload
from src.validator.contracts import StateContract, StateScope, Comparator
from src.validator.registry import StateRegistry
from src.runtime import distributed as dist

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = nn.functional.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerConfig:
    def __init__(self, vocab_size=50304, block_size=128, n_layer=4, n_head=4, n_embd=128, bias=False):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.bias = bias

class SmallTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(0.0),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # tie weights

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

class SmallTransformerWorkload(Workload):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get("batch_size", 8)
        self.lr = config.get("lr", 5e-4)
        self.vocab_size = config.get("vocab_size", 50304)
        self.block_size = config.get("block_size", 64)

    def build_model(self) -> torch.nn.Module:
        cfg = TransformerConfig(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            n_layer=self.config.get("n_layer", 2),
            n_head=self.config.get("n_head", 4),
            n_embd=self.config.get("n_embd", 128)
        )
        model = SmallTransformer(cfg)
        device = torch.device(f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-2)

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    def build_data(self) -> Tuple[DataLoader, DistributedSampler]:
        # Generate a synthetic dataset offline
        torch.manual_seed(1234)
        num_samples = 1000
        data_x = torch.randint(0, self.vocab_size, (num_samples, self.block_size))
        data_y = torch.randint(0, self.vocab_size, (num_samples, self.block_size))
        dataset = TensorDataset(data_x, data_y)
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=self.config.get("seed", 42)
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=0,
            drop_last=True
        )
        return dataloader, sampler

    def train_step(self, model, optimizer, batch, scheduler=None, **kwargs) -> float:
        device = next(model.parameters()).device
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
        return loss.item()

    def evaluate(self, model, dataloader, **kwargs) -> float:
        device = next(model.parameters()).device
        model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if i >= 5: # Fast eval
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                _, loss = model(inputs, targets)
                total_loss += loss.item()
                batches += 1
                
        model.train()
        avg_loss = total_loss / max(1, batches)
        
        if dist.is_initialized():
            loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(loss_tensor)
            avg_loss = loss_tensor.item() / dist.get_world_size()
            
        return avg_loss

    def register_state_contracts(self, registry: StateRegistry, context: Dict[str, Any]) -> None:
        registry.register(
            StateContract(
                name="global_step",
                scope=StateScope.GLOBAL,
                comparator=Comparator.EXACT
            ),
            getter=lambda ctx: ctx["global_step"]
        )

        registry.register(
            StateContract(
                name="model",
                scope=StateScope.REPLICATED,
                comparator=Comparator.ALLCLOSE,
                rtol=1e-4,
                atol=1e-5
            ),
            getter=lambda ctx: {k: v.cpu() for k, v in ctx["model"].state_dict().items()}
        )

        registry.register(
            StateContract(
                name="optimizer",
                scope=StateScope.REPLICATED,
                comparator=Comparator.ALLCLOSE,
                rtol=1e-4,
                atol=1e-5,
                description="Optimizer state. Uses ALLCLOSE for tensor momentum buffers."
            ),
            getter=lambda ctx: ctx["optimizer"].state_dict()
        )

        registry.register(
            StateContract(
                name="scheduler",
                scope=StateScope.REPLICATED,
                comparator=Comparator.EXACT,
                description="Scheduler state must be strictly equal across ranks."
            ),
            getter=lambda ctx: ctx["scheduler"].state_dict()
        )

        registry.register(
            StateContract(
                name="rng.torch_cpu",
                scope=StateScope.PER_RANK,
                comparator=Comparator.EXACT,
                description="PyTorch CPU RNG state. PER_RANK because seeds are rank-offset."
            ),
            getter=lambda ctx: torch.get_rng_state()
        )
