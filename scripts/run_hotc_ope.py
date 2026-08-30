"""Kaggle runner: requires a concrete SAMURAI predictor factory supplied by integration code."""
import argparse,json,platform,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hotc_tracker.data.sequences import group_annotations
from hotc_tracker.tracking.samurai_tracker import resolve_samurai_assets
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--annotations',required=True);p.add_argument('--checkpoint');p.add_argument('--config');p.add_argument('--representation',choices=['robust_band_rgb','pca_rgb'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
 assets=resolve_samurai_assets(checkpoint=a.checkpoint,samurai_config=a.config); groups=group_annotations(a.annotations); out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 # Deliberately fail before inference until the mounted SAMURAI API is verified and bound to SamuraiTracker.
 metadata={'python':platform.python_version(),'representation':a.representation,'data_root':a.data_root,'assets':{k:str(v) for k,v in assets.items()},'sequences':len(groups),'status':'assets_resolved_api_binding_pending','started_at':time.time()};(out/'metadata.json').write_text(json.dumps(metadata,indent=2)); raise RuntimeError('SAMURAI source API binding is required before real HOTC inference; no predictions were generated.')
if __name__=='__main__': main()
