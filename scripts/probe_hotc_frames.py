import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hotc_tracker.data.loaders import read_frame
p=argparse.ArgumentParser();p.add_argument('frames',nargs=3);p.add_argument('--key');a=p.parse_args()
for path in a.frames:
 x=read_frame(path,a.key); sensor=Path(path).name.split('-',1)[0].lower()
 if sensor=='rednir' and x.shape[-1]==16 and not x[...,-1].any():x=x[...,:-1]
 print({'path':path,'shape':x.shape,'dtype':str(x.dtype),'min':float(x.min()),'max':float(x.max()),'number_of_channels':x.shape[-1]})
