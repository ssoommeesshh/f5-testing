"""Seeded wrapper around the official installed F5-TTS Python package.

The official CLI has no --seed option (checked against upstream infer_cli.py on
2026-11-05), so this wrapper sets torch/numpy/random seeds before calling the
same public-package inference utilities used by that CLI.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--ref-audio',required=True); p.add_argument('--ref-text',required=True); p.add_argument('--gen-text',required=True); p.add_argument('--seed',required=True,type=int); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda'); a=p.parse_args()
    import numpy as np, soundfile as sf, torch
    from cached_path import cached_path
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    model_name='F5TTS_v1_Base'
    # These are the same default checkpoint/config locations used by upstream infer_cli.py.
    from importlib.resources import files
    cfg=OmegaConf.load(str(files('f5_tts').joinpath(f'configs/{model_name}.yaml')))
    cls=get_class(f'f5_tts.model.{cfg.model.backbone}')
    ckpt=str(cached_path(f'hf://SWivid/F5-TTS/{model_name}/model_1250000.safetensors'))
    vocoder=load_vocoder('vocos', is_local=False, local_path='../checkpoints/vocos-mel-24khz', device=a.device)
    model=load_model(cls,cfg.model.arch,ckpt,mel_spec_type='vocos',vocab_file='',device=a.device)
    ref_audio,ref_text=preprocess_ref_audio_text(a.ref_audio, Path(a.ref_text).read_text(encoding='utf8'))
    audio,sr,_=infer_process(ref_audio,ref_text,Path(a.gen_text).read_text(encoding='utf8'),model,vocoder,mel_spec_type='vocos',target_rms=0.1,cross_fade_duration=0.15,nfe_step=32,cfg_strength=2.0,sway_sampling_coef=-1.0,speed=1.0,fix_duration=None,device=a.device)
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); sf.write(output,audio,sr)
    output.with_suffix('.generation.json').write_text(json.dumps({'seed':a.seed,'model':model_name,'checkpoint':ckpt,'remove_silence':False,'manual_gaps_inserted':False,'text_unchanged':True,'chunk_boundaries':[]},indent=2))
if __name__ == '__main__': main()
