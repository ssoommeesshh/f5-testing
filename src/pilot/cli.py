from __future__ import annotations
import argparse, csv, json, shutil, subprocess, sys, os, platform
from pathlib import Path
from .text import error_rates
from .timing import Word, gaps
from .report import write_html

ROOT = Path(__file__).resolve().parents[2]

def run(cmd):
    print("+", " ".join(map(str, cmd))); subprocess.run(cmd, check=True)

def manifest(): return json.loads((ROOT / "manifests/dan_barber_2010.json").read_text())

def prepare(a):
    m = manifest(); out = Path(a.workdir); out.mkdir(parents=True, exist_ok=True)
    if a.source == "tedlium":
        if not a.tedlium_root: raise SystemExit("TED-LIUM requires --tedlium-root containing the already obtained legacy/test files; auto-download is intentionally disabled because the verified archive returned 404 and is whole-corpus.")
        root = Path(a.tedlium_root); audio = root / m['primary_source']['expected_paths']['audio']; stm = root / m['primary_source']['expected_paths']['transcript']
        if not audio.exists() or not stm.exists(): raise SystemExit(f"missing expected files: {audio} / {stm}")
        shutil.copy2(stm, out / 'source.stm'); print('Use your local audio path:', audio)
    else:
        print('EXPLICIT FALLBACK: official TED page, not TED-LIUM. Record this in results.')
        run([sys.executable, '-m', 'yt_dlp', '--write-subs', '--sub-langs', 'en', '--convert-subs', 'vtt', '-x', '--audio-format', 'wav', '-o', str(out/'source.%(ext)s'), m['explicit_fallback']['talk_page']])
        vtts=list(out.glob('*.vtt'))
        if not vtts: raise SystemExit('Official TED captions were not fetched; refusing to invent a matching transcript.')
    (out/'runtime_manifest.json').write_text(json.dumps(m, indent=2))

def align(a):
    # WhisperX is maintained and supports en. Alignment is evidence of timings, not textual correctness.
    import whisperx
    device = a.device
    model = whisperx.load_model(a.asr_model, device, language='en')
    audio = whisperx.load_audio(a.audio); result = model.transcribe(audio, batch_size=a.batch_size, language='en')
    amodel, meta = whisperx.load_align_model(language_code='en', device=device)
    aligned = whisperx.align(result['segments'], amodel, meta, audio, device, return_char_alignments=False)
    words=[Word(w['word'], float(w['start']), float(w['end']), w.get('score')) for s in aligned['segments'] for w in s.get('words', []) if 'start' in w and 'end' in w]
    try: candidate=gaps(words, a.gap_threshold)
    except ValueError as e: raise SystemExit(f'FAILED ALIGNMENT: {e}; no timing score produced')
    out=Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf8') as f:
        w=csv.DictWriter(f, fieldnames=words[0].__dict__.keys()); w.writeheader(); w.writerows([x.__dict__ for x in words])
    with out.with_name('candidates.csv').open('w', newline='', encoding='utf8') as f: csv.DictWriter(f, fieldnames=candidate[0].keys()).writeheader(); csv.DictWriter(f, fieldnames=candidate[0].keys()).writerows(candidate)
    (out.with_suffix('.segments.json')).write_text(json.dumps(aligned['segments'], indent=2))
    (out.with_suffix('.asr.json')).write_text(json.dumps(result['segments'], indent=2))

def read_csv(path):
    with Path(path).open(encoding='utf8', newline='') as f: return list(csv.DictReader(f))

def numeric_gap_rows(rows):
    return [{**r, 'after_word_index':int(r['after_word_index']), 'gap_seconds':float(r['gap_seconds'])} for r in rows]

def analyze(a):
    d=Path(a.outdir); d.mkdir(parents=True,exist_ok=True)
    rates=error_rates(Path(a.reference_text).read_text(), Path(a.hypothesis_text).read_text())
    summary={'status':'automated; manual review remains required','asr':rates,'alignment_note':'Word timing is valid only if alignment completed; forced alignment does not prove spoken-word correctness.'}
    if a.reference_breaks and a.generated_breaks:
        from .timing import compare_breaks
        reviewed=read_csv(a.reviewed_breaks) if a.reviewed_breaks else []
        labels={int(r['after_word_index']):r.get('label','') for r in reviewed if r.get('after_word_index','').isdigit()}
        ref=numeric_gap_rows(read_csv(a.reference_breaks)); gen=numeric_gap_rows(read_csv(a.generated_breaks))
        for r in ref: r['reviewed_label']=labels.get(r['after_word_index'],'')
        comparison=compare_breaks(ref, gen, a.tolerance)
        summary['breaks']=comparison
        with (Path(a.outdir)/'per_break.csv').open('w',newline='',encoding='utf8') as f:
            fields=['after_word_index','after_word','before_word','reference_gap_seconds','generated_gap_seconds','reviewed_label','outcome']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            ri={r['after_word_index']:r for r in ref}; gi={r['after_word_index']:r for r in gen}
            for i in sorted(set(ri)|set(gi)):
                label=labels.get(i,''); outcome='matched' if i in comparison['matched'] else ('missing' if i in comparison['missing'] else ('extra' if i in comparison['extra'] else 'unreviewed'))
                w.writerow({'after_word_index':i,'after_word':ri.get(i,gi.get(i,{})).get('after_word',''),'before_word':ri.get(i,gi.get(i,{})).get('before_word',''),'reference_gap_seconds':ri.get(i,{}).get('gap_seconds',''),'generated_gap_seconds':gi.get(i,{}).get('gap_seconds',''),'reviewed_label':label,'outcome':outcome})
    (d/'summary.json').write_text(json.dumps(summary,indent=2)); write_html(summary,d/'report.html')

def generate(a):
    # Official package CLI is invoked rather than reimplementing model inference. Baseline text is untouched.
    if a.mode == 'long': print('Long target will be chunked; output is explicitly reported as multi-chunk, never a single generation.')
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in a.seeds:
        cmd=['f5-tts_infer-cli','--model','F5TTS_v1_Base','--ref_audio',a.reference_audio,'--ref_text',Path(a.reference_text).read_text(),'--gen_text',Path(a.target_text).read_text(),'--seed',str(seed),'--output_dir',str(out)]
        before={str(p) for p in out.rglob('*') if p.is_file()}
        run(cmd)
        after={str(p) for p in out.rglob('*') if p.is_file()}
        records.append({'seed':seed,'command':cmd,'new_files':sorted(after-before),'model':'F5TTS_v1_Base','mode':a.mode,'baseline_text_unchanged':True,'remove_silence':False,'manual_gaps_inserted':False,'automatic_chunk_boundaries':'Read F5 CLI log/output; unavailable boundaries must remain null, not inferred.'})
    record={'reference_audio':a.reference_audio,'reference_transcript':Path(a.reference_text).read_text(),'target_transcript':Path(a.target_text).read_text(),'runs':records,'environment':{'python':sys.version,'platform':platform.platform(),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'pip_freeze':'run `pip freeze > environment.txt` in Kaggle for exact resolved versions'}}
    (out/'generation_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf8')

def main():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(required=True)
    x=sp.add_parser('prepare'); x.add_argument('--source',choices=['tedlium','ted_fallback'],required=True); x.add_argument('--tedlium-root'); x.add_argument('--workdir',default='/kaggle/working/f5-pilot/data'); x.set_defaults(func=prepare)
    x=sp.add_parser('align'); x.add_argument('audio'); x.add_argument('--out',required=True); x.add_argument('--device',default='cuda'); x.add_argument('--asr-model',default='small.en'); x.add_argument('--batch-size',type=int,default=8); x.add_argument('--gap-threshold',type=float,default=.18); x.set_defaults(func=align)
    x=sp.add_parser('generate'); x.add_argument('--reference-audio',required=True); x.add_argument('--reference-text',required=True); x.add_argument('--target-text',required=True); x.add_argument('--output-dir',default='/kaggle/working/f5-pilot/outputs'); x.add_argument('--mode',choices=['short','long'],default='short'); x.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]); x.set_defaults(func=generate)
    x=sp.add_parser('analyze'); x.add_argument('--reference-text',required=True); x.add_argument('--hypothesis-text',required=True); x.add_argument('--reference-breaks'); x.add_argument('--generated-breaks'); x.add_argument('--reviewed-breaks'); x.add_argument('--tolerance',type=float,default=.12); x.add_argument('--outdir',default='/kaggle/working/f5-pilot/report'); x.set_defaults(func=analyze)
    a=p.parse_args(); a.func(a)
if __name__ == '__main__': main()
