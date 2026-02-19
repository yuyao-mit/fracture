
from typing import Dict, Type, Tuple
import torch.nn as nn
from .uno import UNO
from .codano import CODANO
from .fno import FNO
from .rno import RNO


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "fno": FNO,
    "uno": UNO,
    "codano": CODANO,
    "rno": RNO,
}

def build_model(input_shape: Tuple, output_shape: Tuple, model_name: str):

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
