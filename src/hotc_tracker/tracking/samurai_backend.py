"""Binding for the supplied SAM2/SAMURAI video predictor (not a replacement model)."""
from __future__ import annotations
import importlib, sys
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from .samurai_tracker import resolve_samurai_assets

class SamuraiVideoPredictor:
    def __init__(self, predictor, device='cuda', dtype='float16'):
        self.predictor=predictor; self.device=device; self.dtype=dtype; self.state=None; self._outputs=None
    def initialize(self, video_dir, xywh):
        self.state=self.predictor.init_state(video_path=str(video_dir))
        x,y,w,h=map(float,xywh); box=np.array([x,y,x+w,y+h],dtype=np.float32)
        # SAM2's add_new_points_or_box accepts pixel xyxy boxes.
        self.predictor.add_new_points_or_box(self.state, frame_idx=0, obj_id=1, box=box)
        self._outputs=iter(self.predictor.propagate_in_video(self.state))
    def next_mask(self):
        frame_idx, object_ids, logits=next(self._outputs)
        # Propagation includes the prompted first frame; it is already supplied
        # by HOTC ground-truth initialization and must not be emitted as frame 2.
        while frame_idx == 0:
            frame_idx, object_ids, logits=next(self._outputs)
        index=list(object_ids).index(1)
        return int(frame_idx), (logits[index] > 0).detach().cpu().numpy().squeeze()
    def reset(self):
        if self.state is not None: self.predictor.reset_state(self.state)
        self.state=None; self._outputs=None

def build_samurai_predictor(checkpoint=None, config=None, *, source_root=None, device='cuda', dtype='float16'):
    """Build exact mounted `build_sam2_video_predictor`; validates P100-safe dtype."""
    assets=resolve_samurai_assets(checkpoint=checkpoint, samurai_config=config, source_root=source_root)
    if dtype == 'bfloat16': raise ValueError('bfloat16 is disabled: Tesla P100 requires float16')
    source=str(assets['source_root']/'sam2')
    if source not in sys.path: sys.path.insert(0,source)
    try: builder=importlib.import_module('sam2.build_sam').build_sam2_video_predictor
    except (ImportError, AttributeError) as exc: raise RuntimeError('Mounted SAMURAI source does not expose sam2.build_sam.build_sam2_video_predictor') from exc
    predictor=builder(str(assets['samurai_config']), str(assets['checkpoint']), device=device)
    return SamuraiVideoPredictor(predictor, device, dtype)
