import os
import torch.distributed as dist

def is_initialized() -> bool:
    return dist.is_initialized()

def init_process_group(backend: str = "gloo"):
    """Initialize the distributed process group."""
    if not is_initialized():
        dist.init_process_group(backend=backend)

def destroy_process_group():
    """Cleanup the distributed process group."""
    if is_initialized():
        dist.destroy_process_group()

def get_rank() -> int:
    """Get the global rank of the current process."""
    if not is_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    """Get the world size of the distributed run."""
    if not is_initialized():
        return 1
    return dist.get_world_size()

def barrier():
    """Synchronize all processes."""
    if is_initialized():
        dist.barrier()

def gather_object(obj):
    """
    Gather an object from all ranks to rank 0.
    Returns a list of objects on rank 0, and None on other ranks.
    """
    if not is_initialized():
        return [obj]
        
    world_size = get_world_size()
    rank = get_rank()
    
    # We must use all_gather_object as gather_object requires specific setup in some backends
    # or just broadcast/gather if gloo supports it. `all_gather_object` is generally safest.
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered

def broadcast_object(obj, src=0):
    """
    Broadcast an object from src to all other ranks.
    """
    if not is_initialized():
        return obj
        
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]
