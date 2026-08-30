"""Kaggle-only model/source compatibility check; does not require raw frames."""
import argparse, hashlib, inspect, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hotc_tracker.tracking.samurai_tracker import resolve_samurai_assets

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--config');p.add_argument('--source-root');p.add_argument('--build',action='store_true');a=p.parse_args(); assets=resolve_samurai_assets(a.checkpoint,a.config,source_root=a.source_root)
 sys.path.insert(0,str(assets['source_root']/'sam2')); from sam2.build_sam import build_sam2_video_predictor
 print('SAMURAI IMPORT: OK'); print('BUILDER SIGNATURE:',inspect.signature(build_sam2_video_predictor)); print('CHECKPOINT:',assets['checkpoint'],assets['checkpoint'].stat().st_size,sha256(assets['checkpoint'])); print('CONFIG:',assets['samurai_config']); print(Path(assets['samurai_config']).read_text(errors='replace')[:800])
 import torch; print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none','dtype: float16')
 if a.build:
  from hotc_tracker.tracking.samurai_backend import build_samurai_predictor
  model=build_samurai_predictor(assets['checkpoint'],assets['samurai_config'],source_root=assets['source_root']); print('PREDICTOR BUILD: OK',type(model.predictor)); del model; torch.cuda.empty_cache()
if __name__=='__main__': main()
