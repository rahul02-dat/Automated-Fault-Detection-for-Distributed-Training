from .common import Workload
from .tiny_transformer import TinyTransformerWorkload
from .resnet import ResNetWorkload
from .small_transformer import SmallTransformerWorkload

__all__ = ["Workload", "TinyTransformerWorkload", "ResNetWorkload", "SmallTransformerWorkload"]
