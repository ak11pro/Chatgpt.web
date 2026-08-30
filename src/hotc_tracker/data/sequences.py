"""HOTC annotation identifier parsing and sequence grouping."""
from dataclasses import dataclass
import csv
from pathlib import Path

@dataclass(frozen=True)
class FrameId:
    sensor: str; sequence: str; frame: int
    @property
    def key(self): return f"{self.sensor}-{self.sequence}"

def parse_hotc_id(value: str) -> FrameId:
    stem, sep, token = value.rpartition("_")
    if not sep or not token.isdigit() or not stem: raise ValueError(f"invalid HOTC ID: {value!r}")
    sensor, sep, sequence = stem.partition("-")
    if not sep or not sensor or not sequence: raise ValueError(f"missing sensor/sequence in ID: {value!r}")
    return FrameId(sensor.lower(), sequence, int(token))

def group_annotations(path: str | Path):
    groups = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            parsed = parse_hotc_id(row["ID"]); groups.setdefault(parsed.key, []).append((parsed, row))
    for rows in groups.values():
        rows.sort(key=lambda item:item[0].frame)
        frames=[x[0].frame for x in rows]
        if len(frames)!=len(set(frames)): raise ValueError("duplicate frame indices in sequence")
    return groups
