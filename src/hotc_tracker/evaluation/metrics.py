"""OPE metrics. Success curve uses IoU thresholds 0.00..1.00 inclusive; AUC is trapezoidal integral."""
import numpy as np
def iou(p,t):
 p=np.asarray(p,float);t=np.asarray(t,float); end=np.minimum(p[:,:2]+p[:,2:],t[:,:2]+t[:,2:]); inter=np.prod(np.maximum(end-np.maximum(p[:,:2],t[:,:2]),0),1); union=np.prod(p[:,2:],1)+np.prod(t[:,2:],1)-inter;return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
def ope_metrics(predictions,ground_truth):
 p=np.asarray(predictions,float);t=np.asarray(ground_truth,float)
 if p.shape!=t.shape or p.ndim!=2 or p.shape[1]!=4: raise ValueError("predictions and ground_truth must have shape (n, 4)")
 if not len(p): raise ValueError("no box pairs")
 v=iou(p,t); err=np.linalg.norm(p[:,:2]+p[:,2:]/2-t[:,:2]-t[:,2:]/2,axis=1); th=np.linspace(0,1,101); curve=np.array([(v>=x).mean() for x in th]); return {"mean_iou":float(v.mean()),"precision_at_20":float((err<=20).mean()),"success_auc":float(np.trapezoid(curve,th)),"success_curve":curve.tolist(),"frames":float(len(p))}
