import numpy as np
def xywh_to_xyxy(box): x,y,w,h=map(float,box); return np.array([x,y,x+w,y+h])
def xyxy_to_xywh(box): x1,y1,x2,y2=map(float,box); return np.array([x1,y1,x2-x1,y2-y1])
def clip_xywh(box, image_shape):
    h,w=image_shape[:2]; x,y,bw,bh=map(float,box); x=max(0,min(x,w-1)); y=max(0,min(y,h-1)); bw=max(1,min(bw,w-x)); bh=max(1,min(bh,h-y)); return np.rint([x,y,bw,bh]).astype(int)
def mask_to_xywh(mask):
    ys,xs=np.nonzero(np.asarray(mask))
    if not len(xs): raise ValueError("degenerate/empty mask")
    return np.array([xs.min(),ys.min(),xs.max()-xs.min()+1,ys.max()-ys.min()+1],dtype=int)
