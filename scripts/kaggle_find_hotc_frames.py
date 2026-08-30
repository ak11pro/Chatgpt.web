import argparse, os
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='/kaggle/input');p.add_argument('--limit',type=int,default=8);a=p.parse_args(); groups=defaultdict(lambda:[0,0,[]])
for root,_,files in os.walk(a.root):
 for name in files:
  x=Path(name); ext=x.suffix.lower()
  if ext in {'.npy','.npz','.mat','.h5','.hdf5','.tif','.tiff','.png','.jpg','.jpeg'}:
   value=groups[(root,ext)];value[0]+=1
   try:value[1]+= (Path(root)/name).stat().st_size
   except OSError:pass
   if len(value[2])<a.limit:value[2].append(name)
for (path,ext),(count,size,examples) in sorted(groups.items(),key=lambda kv:-kv[1][1]): print({'path':path,'extension':ext,'count':count,'total_size':size,'examples':examples})
