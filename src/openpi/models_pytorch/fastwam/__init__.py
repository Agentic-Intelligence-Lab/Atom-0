"""FastWAM (World-Action Model) PyTorch package adapted for Atom-0 / openpi."""

from openpi.models_pytorch.fastwam.factory import create_fastwam
from openpi.models_pytorch.fastwam.wan22.fastwam import FastWAM

__all__ = ["FastWAM", "create_fastwam"]
