"""Small, dependency-free caption/STM parsing and clip-plan validation."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str

def _seconds(value: str) -> float:
    h, m, s = value.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def read_vtt(path: Path) -> list[Cue]:
    blocks=re.split(r'\n\s*\n', path.read_text(encoding='utf8').replace('\r\n','\n'))
    out=[]
    for block in blocks:
        lines=[x.strip() for x in block.splitlines() if x.strip()]
        timing=next((x for x in lines if '-->' in x), None)
        if not timing: continue
        left,right=timing.split('-->',1); text=' '.join(x for x in lines[lines.index(timing)+1:] if not x.startswith('NOTE'))
        text=re.sub(r'<[^>]+>', '', text).strip()
        if text: out.append(Cue(_seconds(left.strip()), _seconds(right.strip().split()[0]), text))
    return out

def read_stm(path: Path) -> list[Cue]:
    out=[]
    for line in path.read_text(encoding='utf8').splitlines():
        p=line.split(maxsplit=6)
        if len(p)==7: out.append(Cue(float(p[3]),float(p[4]),p[6]))
    return out

def select(cues: list[Cue], start: float, end: float) -> str:
    parts=[c.text for c in cues if c.start < end and c.end > start]
    # Caption providers occasionally repeat an identical cue on adjacent updates.
    kept=[]
    for part in parts:
        if not kept or part != kept[-1]: kept.append(part)
    text=re.sub(r'\s+', ' ', ' '.join(kept)).strip()
    if not text: raise ValueError(f'no transcript cues overlap {start}-{end}s')
    return text

def validate_plan(reference: tuple[float,float], targets: dict[str,tuple[float,float]]) -> None:
    rs,re=reference
    if not 5 <= re-rs <= 10: raise ValueError('reference duration must be 5–10 seconds')
    for name,(s,e) in targets.items():
        if s >= e: raise ValueError(f'{name}: invalid window')
        if max(rs,s) < min(re,e): raise ValueError(f'{name}: overlaps reference window')
