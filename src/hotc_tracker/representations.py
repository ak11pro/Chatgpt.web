"""Deterministic HSI spatial representations; callers retain the original cube."""
import numpy as np
SENSOR_BANDS={"vis":16,"nir":25,"rednir":15}
def validate_cube(cube, expected_bands=None):
    a=np.asarray(cube)
    if a.ndim!=3 or a.shape[-1]<3: raise ValueError("cube must have shape (H, W, C>=3)")
    if expected_bands is not None and a.shape[-1]!=expected_bands: raise ValueError(f"expected {expected_bands} bands, got {a.shape[-1]}")
    return a.astype(np.float32,copy=False)
def robust_band_rgb(cube,bands=None):
    a=validate_cube(cube); bands=tuple(bands) if bands is not None else tuple(np.linspace(0,a.shape[-1]-1,3,dtype=int))
    if len(bands)!=3 or min(bands)<0 or max(bands)>=a.shape[-1]: raise ValueError("bands must be three valid indices")
    v=a[...,bands]; lo=np.percentile(v,1,(0,1),keepdims=True); hi=np.percentile(v,99,(0,1),keepdims=True)
    return np.rint(np.clip((v-lo)/np.maximum(hi-lo,1e-6),0,1)*255).astype(np.uint8)
def pca_rgb(cube):
    a=validate_cube(cube); h,w,c=a.shape; x=a.reshape(-1,c); x=x-x.mean(0); _,_,vt=np.linalg.svd(x,full_matrices=False); return robust_band_rgb((x@vt[:3].T).reshape(h,w,3),(0,1,2))
def representation(cube, kind, sensor=None, bands_by_sensor=None):
    if kind=="pca_rgb": return pca_rgb(cube)
    if kind=="robust_band_rgb": return robust_band_rgb(cube,(bands_by_sensor or {}).get((sensor or "").lower()))
    raise ValueError(f"unknown representation: {kind}")
