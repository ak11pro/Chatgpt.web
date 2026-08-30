"""Run one or more mounted HOTC sequences through SAMURAI video propagation."""
import argparse,csv,json,platform,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from hotc_tracker.data.loaders import DirectoryFrameSource
from hotc_tracker.data.sequences import group_annotations
from hotc_tracker.evaluation import ope_metrics
from hotc_tracker.representations import representation, SENSOR_BANDS
from hotc_tracker.tracking.box_utils import clip_xywh, mask_to_xywh
from hotc_tracker.tracking.samurai_backend import build_samurai_predictor
from hotc_tracker.tracking.samurai_tracker import SamuraiTracker

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True);p.add_argument('--annotations',required=True);p.add_argument('--checkpoint');p.add_argument('--config');p.add_argument('--source-root');p.add_argument('--representation',choices=['robust_band_rgb','pca_rgb'],required=True);p.add_argument('--output',required=True);p.add_argument('--sequences',nargs='*');p.add_argument('--frames',type=int);p.add_argument('--extension',default='.npy');p.add_argument('--frame-key');a=p.parse_args()
 groups=group_annotations(a.annotations); chosen=a.sequences or list(groups); out=Path(a.output);out.mkdir(parents=True,exist_ok=True); source=DirectoryFrameSource(a.data_root,a.extension,a.frame_key,SENSOR_BANDS); all_preds=[]; per=[]; started=time.time()
 backend=build_samurai_predictor(a.checkpoint,a.config,source_root=a.source_root)
 for key in chosen:
  rows=groups[key][:a.frames] if a.frames else groups[key]; sensor=rows[0][0].sensor; gt=np.array([[float(r[k]) for k in ('x','y','width','height')] for _,r in rows]);
  with tempfile.TemporaryDirectory(prefix='hotc_rgb_') as d:
   d=Path(d); cubes=[]
   for parsed,_ in rows:
    cube=source.frame(key,parsed.frame); rgb=representation(cube,a.representation,sensor); Image.fromarray(rgb).save(d/f'{parsed.frame-rows[0][0].frame:05d}.jpg'); cubes.append(cube.shape)
   # SAM2 sorts frame image names; use contiguous names above and first GT prompt.
   tracker=SamuraiTracker(backend); tracker.initialize(d,gt[0]); preds=[clip_xywh(gt[0],cubes[0])]; fallbacks=[]
   for index in range(1,len(rows)):
    try: preds.append(clip_xywh(tracker.track(None),cubes[index]))
    except (StopIteration,ValueError) as exc: preds.append(preds[-1]); fallbacks.append({'frame':rows[index][0].frame,'reason':type(exc).__name__})
  tracker.reset(); metrics=ope_metrics(np.asarray(preds),gt); per.append({'sequence':key,'sensor':sensor,**{k:v for k,v in metrics.items() if k!='success_curve'},'fallback_count':len(fallbacks)}); all_preds += [{'ID':r['ID'],'x':b[0],'y':b[1],'width':b[2],'height':b[3]} for (_,r),b in zip(rows,preds)]
  print({'sequence':key,'sensor':sensor,'frames':len(rows),'hsi_shape':cubes[0],'first_box':preds[0].tolist(),'last_box':preds[-1].tolist(),**per[-1]})
 with (out/'predictions.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['ID','x','y','width','height']);w.writeheader();w.writerows(all_preds)
 with (out/'per_sequence.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=per[0].keys());w.writeheader();w.writerows(per)
 aggregate={k:float(np.mean([r[k] for r in per])) for k in ('mean_iou','precision_at_20','success_auc','fallback_count')};(out/'aggregate.json').write_text(json.dumps(aggregate,indent=2));(out/'metadata.json').write_text(json.dumps({'python':platform.python_version(),'representation':a.representation,'runtime_seconds':time.time()-started,'sequences':chosen},indent=2))
if __name__=='__main__': main()
