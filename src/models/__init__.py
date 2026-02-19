from typing import Dict, Type, Tuple
import torch.nn as nn
# Model Factory
from .convlstm import ConvLSTM
from .predrnn_v2 import PredRNNv2
from .phydnet import PhyDNet
from .simvp import SimVP
from .stgcn import STGCN
from .fno import FNO
from .voxelflow import VoxelFlow
from .mlp import MLP
from .gwnet import GWNet
from .agcrn import AGCRN
from .trajgru import TrajGRU
from .uno import UNO
from .patchtst import PatchTST
from .autoformer import Autoformer
from .informer import Informer
from .gkn import GKN
from .codano import CODANO

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    # RNN / Video
    "convlstm": ConvLSTM,
    "predrnn_v2": PredRNNv2,
    "trajgru": TrajGRU,
    "phydnet": PhyDNet,
    "simvp": SimVP,
    "voxelflow": VoxelFlow,

    # GNN
    "stgcn": STGCN,
    "gwnet": GWNet,
    "agcrn": AGCRN,
    "gkn": GKN,

    # Neural Operator
    "fno": FNO,
    "uno": UNO,
    "codano": CODANO,

    # Transformer
    "patchtst": PatchTST,
    "autoformer": Autoformer,
    "informer": Informer,

    # Baseline
    "mlp": MLP,
}

def build_model(input_shape: Tuple, output_shape: Tuple, model_name: str):
    """
    Build model with a unified interface:
        input  : (B, T, C, H, W)
        output : (B, T, C, H, W)

    All channel merging / reshaping is handled INSIDE models.
    """
    if len(input_shape) != 5 or len(output_shape) != 5:
        raise ValueError(f"Expected input/output shape (B, T, C, H, W), got {input_shape} and {output_shape}")

    model_name = model_name.lower()

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    model_class = MODEL_REGISTRY[model_name]

    model = model_class(input_shape=input_shape, output_shape=output_shape)

    return model


__all__ = [
    "build_model",
    "MODEL_REGISTRY",
]
