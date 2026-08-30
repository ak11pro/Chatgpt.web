"""Format-specific HSI readers; optional scientific dependencies are lazy."""
from pathlib import Path
import numpy as np
from .frames import FrameSource

def validate_frame(frame, expected_bands=None):
    frame=np.asarray(frame)
    if frame.ndim==2: frame=frame[...,None]
    if frame.ndim!=3: raise ValueError(f"frame must be HWC, got {frame.shape}")
    if expected_bands and frame.shape[-1]!=expected_bands: raise ValueError(f"expected {expected_bands} bands, got {frame.shape[-1]}")
    return frame

def read_frame(path, key=None, expected_bands=None):
    path=Path(path); suffix=path.suffix.lower()
    if suffix==".npy": value=np.load(path, allow_pickle=False)
    elif suffix==".npz":
        with np.load(path, allow_pickle=False) as z:
            name=key or (z.files[0] if len(z.files)==1 else None)
            if name is None or name not in z: raise ValueError("NPZ requires an explicit key when it contains multiple arrays")
            value=z[name]
    elif suffix in {".h5", ".hdf5"}:
        import h5py
        if not key: raise ValueError("HDF5 requires an explicit dataset key")
        with h5py.File(path) as f: value=f[key][...]
    elif suffix==".mat":
        from scipy.io import loadmat
        if not key: raise ValueError("MAT requires an explicit variable key")
        value=loadmat(path)[key]
    else: raise ValueError(f"unsupported cube format: {suffix}")
    return validate_frame(value, expected_bands)

class DirectoryFrameSource(FrameSource):
    def __init__(self, root, extension=".npy", key=None, sensor_bands=None): self.root=Path(root); self.extension=extension; self.key=key; self.sensor_bands=sensor_bands or {}
    def frame(self, sequence, frame_index):
        sensor=sequence.split("-",1)[0].lower(); path=self.root/sequence/f"{frame_index}{self.extension}"
        return read_frame(path, self.key, self.sensor_bands.get(sensor))
