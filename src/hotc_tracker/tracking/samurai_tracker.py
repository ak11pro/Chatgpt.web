"""Stable adapter around an injected SAMURAI-compatible temporal predictor.
The Kaggle source is resolved lazily because it is not shipped in this repository.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
from .box_utils import mask_to_xywh
KAGGLE_CHECKPOINT="/kaggle/input/datasets/akhileshgodugu/hotc-2026-offline-models/dataset2/sam2.1_hiera_large.pt"
KAGGLE_SOURCE="/kaggle/input/datasets/akhileshgodugu/hotc-samurai-code/samurai-master"
def resolve_asset(explicit=None, env=None, candidates=(), label="asset"):
    values=[explicit, os.getenv(env) if env else None, *candidates]
    for value in values:
        if value and Path(value).exists(): return Path(value)
    raise FileNotFoundError(f"{label} not found; set {env or 'an explicit path'}")
def resolve_samurai_assets(checkpoint=None, sam2_config=None, samurai_config=None, source_root=None):
    root=resolve_asset(source_root,"HOTC_SAMURAI_ROOT",[KAGGLE_SOURCE],"SAMURAI source")
    return {"source_root":root,"checkpoint":resolve_asset(checkpoint,"HOTC_SAM2_CHECKPOINT",[KAGGLE_CHECKPOINT],"SAM2 checkpoint"),"sam2_config":resolve_asset(sam2_config,"HOTC_SAM2_CONFIG",[root/"sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"],"SAM2 config"),"samurai_config":resolve_asset(samurai_config,"HOTC_SAMURAI_CONFIG",[root/"sam2/sam2/configs/samurai/sam2.1_hiera_l.yaml"],"SAMURAI config")}
@dataclass
class SamuraiState: box: np.ndarray|None=None
class SamuraiTracker:
    """Predictor must implement initialize(rgb, xywh), track(rgb)->binary mask or xywh, reset()."""
    def __init__(self, predictor): self.predictor=predictor; self.state=SamuraiState()
    def initialize(self, frame, xywh): self.state.box=np.asarray(xywh,float); self.predictor.initialize(frame, self.state.box); return self.state.box.copy()
    def track(self, frame):
        output=self.predictor.track(frame); arr=np.asarray(output)
        self.state.box=mask_to_xywh(arr) if arr.ndim==2 else arr.astype(float)
        return self.state.box.copy()
    def reset(self): self.predictor.reset(); self.state=SamuraiState()
