"""Deterministic spectral primitives; fusion is intentionally not implemented."""
import numpy as np
def normalized_signature(cube, mask):
    values=np.asarray(cube)[np.asarray(mask,dtype=bool)]
    if not len(values): raise ValueError("empty spectral patch")
    s=values.mean(0); return s/(np.linalg.norm(s)+1e-12)
def cosine_similarity(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def spectral_angle(a,b): return float(np.arccos(np.clip(cosine_similarity(a,b),-1,1)))
