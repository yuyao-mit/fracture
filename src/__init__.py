
from typing import Dict, Type
import torch.nn as nn


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


# ============================
# Registry
# ============================

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    # RNN
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

    # Misc
    "mlp": MLP,
    
}

def build_model(cfg):
    """
    Expected config structure (flexible):
        cfg.model.name : str
        cfg.model.*    : model-specific kwargs

    This function does NOT:
        - move model to device
        - load checkpoint
        - create optimizer / loss
    """

    if not hasattr(cfg, "model"):
        raise AttributeError("Config has no attribute 'model'")

    model_cfg = cfg.model

    if not hasattr(model_cfg, "name"):
        raise AttributeError("cfg.model has no attribute 'name'")

    model_name = model_cfg.name.lower()

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    model_class = MODEL_REGISTRY[model_name]

    # Remove 'name' from kwargs before passing to model
    kwargs = {
        k: v for k, v in vars(model_cfg).items()
        if k != "name"
    }

    model = model_class(**kwargs)

    return model


__all__ = [
    "build_model",
    "MODEL_REGISTRY",
]
