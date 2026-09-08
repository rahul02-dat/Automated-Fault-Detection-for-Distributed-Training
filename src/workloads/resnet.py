import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from torchvision.models.resnet import resnet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

from src.workloads.common import Workload
from src.validator.contracts import StateContract, StateScope, Comparator
from src.validator.registry import StateRegistry
from src.runtime import distributed as dist

class ResNetWorkload(Workload):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.lr = config.get("lr", 0.001)

    def build_model(self) -> torch.nn.Module:
        # Build a small resnet (ResNet18) and modify the first conv/fc for CIFAR-10
        model = resnet18(num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        
        # move to device
        device = torch.device(f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    def build_data(self) -> Tuple[DataLoader, DistributedSampler]:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Ensure only rank 0 downloads the dataset to prevent race conditions
        if dist.get_rank() == 0:
            datasets.CIFAR10(root='./data', train=True, download=True)
            
        if dist.is_initialized():
            dist.barrier()
            
        dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
        
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

    def train_step(self, model, optimizer, batch, scheduler=None, step=None, **kwargs) -> float:
        device = next(model.parameters()).device
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        return loss.item()

    def evaluate(self, model, dataloader) -> float:
        # A simple evaluation loop (over just a few batches to save time in tests)
        device = next(model.parameters()).device
        model.eval()
        total_loss = 0.0
        batches = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                if i >= 5: # Only run 5 batches for fast smoke tests
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)
                total_loss += loss.item()
                batches += 1
                
        model.train()
        avg_loss = total_loss / max(1, batches)
        
        # All reduce the loss
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
                comparator=Comparator.CUSTOM
            ),
            getter=lambda ctx: ctx["optimizer"].state_dict()
        )

        registry.register(
            StateContract(
                name="scheduler",
                scope=StateScope.REPLICATED,
                comparator=Comparator.CUSTOM
            ),
            getter=lambda ctx: ctx["scheduler"].state_dict()
        )

        registry.register(
            StateContract(
                name="rng",
                scope=StateScope.PER_RANK,
                comparator=Comparator.EXACT
            ),
            getter=lambda ctx: ctx["rng"]
        )
