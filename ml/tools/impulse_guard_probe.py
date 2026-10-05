"""Cheap front-end sanity probe: normal speech should rarely trigger; impulses should trigger."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import sys
HERE=Path(__file__).resolve(); ML=HERE.parents[1]; sys.path.insert(0,str(ML))
from front_end import MicFrontEnd  # noqa: E402


def read(path):
    x,sr=sf.read(path,dtype='float32')
    if x.ndim>1: x=x.mean(axis=1)
    if sr!=16000: raise SystemExit(f'{path}: expected 16 kHz')
    return x


def run(x):
    fe=MicFrontEnd(); _,_,tr=fe.process_audio(x,np.zeros_like(x)); frames=max(1,int(np.ceil(len(x)/160)))
    return tr, frames

p=argparse.ArgumentParser(); p.add_argument('--speech', required=True); p.add_argument('--impulse'); args=p.parse_args()
sp=read(args.speech); tr,fr=run(sp); print(f'speech: triggers={tr}/{fr} ({100*tr/fr:.2f}%)')
if args.impulse:
    im=read(args.impulse); tr,fr=run(im); print(f'impulse: triggers={tr}/{fr} ({100*tr/fr:.2f}%)')
