from __future__ import annotations
import argparse, csv, json, shutil, subprocess, sys
from types import SimpleNamespace
from pathlib import Path
from .media import read_stm, read_vtt, select, validate_plan
from .report import write_html
from .text import error_rates
from .timing import Word, gaps, compare_breaks

ROOT=Path(__file__).resolve().parents[2]
def run(cmd): print('+',' '.join(map(str,cmd))); subprocess.run(cmd,check=True)
def manifest(): return json.loads((ROOT/'manifests/dan_barber_2010.json').read_text())
def cut(source,target,start,end): run(['ffmpeg','-y','-ss',str(start),'-to',str(end),'-i',str(source),'-ac','1','-ar','24000',str(target)])

def prepare(a):
    m=manifest(); out=Path(a.workdir); out.mkdir(parents=True,exist_ok=True); w=m['planned_windows_seconds']; ref=tuple(w['reference']); targets={'short':tuple(w['target_short']),'long':tuple(w['target_long'])}; validate_plan(ref,targets)
    if a.source=='tedlium':
        if not a.tedlium_root: raise SystemExit('TED-LIUM needs --tedlium-root; automatic whole-corpus download is disabled.')
        root=Path(a.tedlium_root); source=root/m['primary_source']['expected_paths']['audio']; transcript=root/m['primary_source']['expected_paths']['transcript']; cues=read_stm(transcript)
    else:
        print('EXPLICIT FALLBACK: official TED source, not TED-LIUM.')
        run([sys.executable,'-m','yt_dlp','--write-subs','--sub-langs','en','--convert-subs','vtt','-x','--audio-format','wav','-o',str(out/'source.%(ext)s'),m['explicit_fallback']['talk_page']])
        source=next(iter(out.glob('*.wav')),None); transcript=next(iter(out.glob('*.vtt')),None)
        if not source or not transcript: raise SystemExit('TED audio or English VTT was unavailable; refusing to invent a matching transcript.')
        cues=read_vtt(transcript)
    if not source.exists() or not transcript.exists(): raise SystemExit('missing source media/transcript')
    cut(source,out/'ref.wav',*ref); (out/'ref.txt').write_text(select(cues,*ref),encoding='utf8')
    for name,win in targets.items():
        cut(source,out/f'original-target-{name}.wav',*win); (out/f'target-{name}.txt').write_text(select(cues,*win),encoding='utf8')
    shutil.copy2(out/'original-target-long.wav',out/'original-target.wav')
    runtime={**m,'source_used':a.source,'artifacts':{k:str(out/k) for k in ['ref.wav','ref.txt','target-short.txt','target-long.txt','original-target-short.wav','original-target-long.wav','original-target.wav']},'windows':{'reference':ref,'targets':targets}}
    (out/'runtime_manifest.json').write_text(json.dumps(runtime,indent=2),encoding='utf8')

def align(a):
    """Known-text forced alignment plus separate ASR; ASR is never timing ground truth."""
    import whisperx
    known=Path(a.transcript).read_text(encoding='utf8').strip()
    if not known: raise SystemExit('known transcript is empty')
    device=a.device; audio=whisperx.load_audio(a.audio); asr=whisperx.load_model(a.asr_model,device,language='en').transcribe(audio,batch_size=a.batch_size,language='en'); amodel,meta=whisperx.load_align_model(language_code='en',device=device)
    aligned=whisperx.align([{'start':0.0,'end':len(audio)/16000.0,'text':known}],amodel,meta,audio,device,return_char_alignments=False)
    words=[Word(w['word'],float(w['start']),float(w['end']),w.get('score')) for s in aligned['segments'] for w in s.get('words',[]) if 'start' in w and 'end' in w]
    try: candidates=gaps(words,a.gap_threshold)
    except ValueError as e: raise SystemExit(f'FAILED KNOWN-TEXT ALIGNMENT: {e}; no timing score produced')
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf8') as f:
        wr=csv.DictWriter(f,fieldnames=['word','start','end','score']); wr.writeheader(); wr.writerows([x.__dict__ for x in words])
    with out.with_name(out.stem+'_breaks.csv').open('w',newline='',encoding='utf8') as f:
        wr=csv.DictWriter(f,fieldnames=candidates[0].keys()); wr.writeheader(); wr.writerows(candidates)
    out.with_suffix('.asr.txt').write_text(' '.join(s.get('text','') for s in asr['segments']).strip(),encoding='utf8'); out.with_suffix('.asr.json').write_text(json.dumps(asr['segments'],indent=2),encoding='utf8'); out.with_suffix('.known_alignment.json').write_text(json.dumps(aligned['segments'],indent=2),encoding='utf8')

def rows(path):
    with Path(path).open(newline='',encoding='utf8') as f: return list(csv.DictReader(f))
def analyze(a):
    d=Path(a.outdir); d.mkdir(parents=True,exist_ok=True); target=Path(a.target_text).read_text(); hypo=Path(a.asr_text).read_text(); summary={'status':'automated; manual review required','asr':error_rates(target,hypo),'label':'Relative to this speaker’s recording, not objectively incorrect delivery.'}
    ref=rows(a.reference_breaks); gen=rows(a.generated_breaks); labels={int(x['after_word_index']):x.get('label','') for x in rows(a.reviewed_breaks) if x.get('after_word_index','').isdigit()}
    def cast(xs): return [{**x,'after_word_index':int(x['after_word_index']),'gap_seconds':float(x['gap_seconds']),'reviewed_label':labels.get(int(x['after_word_index']),'')} for x in xs]
    comparison=compare_breaks(cast(ref),cast(gen),a.tolerance); summary['breaks']=comparison
    with (d/'per_break.csv').open('w',newline='',encoding='utf8') as f:
        fields=['after_word_index','after_word','before_word','reference_gap_seconds','generated_gap_seconds','reviewed_label','outcome']; wr=csv.DictWriter(f,fieldnames=fields); wr.writeheader(); ri={int(x['after_word_index']):x for x in ref}; gi={int(x['after_word_index']):x for x in gen}
        for i in sorted(set(ri)|set(gi)):
            outcome='matched' if i in comparison['matched'] else 'missing' if i in comparison['missing'] else 'extra' if i in comparison['extra'] else 'unreviewed'; v=ri.get(i,gi.get(i,{})); wr.writerow({'after_word_index':i,'after_word':v.get('after_word',''),'before_word':v.get('before_word',''),'reference_gap_seconds':ri.get(i,{}).get('gap_seconds',''),'generated_gap_seconds':gi.get(i,{}).get('gap_seconds',''),'reviewed_label':labels.get(i,''),'outcome':outcome})
    (d/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8'); write_html(summary,d/'report.html',a.original_audio,a.generated_audio)

def generate(a):
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for seed in a.seeds:
        run([sys.executable,'-m','pilot.f5_runner','--ref-audio',a.reference_audio,'--ref-text',a.reference_text,'--gen-text',a.target_text,'--seed',str(seed),'--output',str(out/f'{a.mode}-seed{seed}.wav'),'--device',a.device])
    (out/'generation_manifest.json').write_text(json.dumps({'mode':a.mode,'seeds':a.seeds,'model':'F5TTS_v1_Base','baseline':'text unchanged; no remove-silence or inserted gaps','chunk_note':'Each WAV is one seeded call. Long text is not labelled a single long generation; internal F5 boundaries are not inferred.'},indent=2))

def full_run(a):
    """The Kaggle one-command chain; it deliberately stops on any failed stage."""
    data=Path(a.workdir); prepare(SimpleNamespace(source=a.source,tedlium_root=a.tedlium_root,workdir=str(data)))
    mode=a.mode; target=data/f'target-{mode}.txt'; original=data/f'original-target-{mode}.wav'; out=Path(a.output_dir)
    generate(SimpleNamespace(reference_audio=str(data/'ref.wav'),reference_text=str(data/'ref.txt'),target_text=str(target),mode=mode,output_dir=str(out),seeds=a.seeds,device=a.device))
    align(SimpleNamespace(audio=str(original),transcript=str(target),out=str(out/'original_words.csv'),device=a.device,asr_model=a.asr_model,batch_size=a.batch_size,gap_threshold=a.gap_threshold))
    review=a.reviewed_breaks or str(ROOT/'annotations/reviewed_breaks.csv')
    for seed in a.seeds:
        wav=out/f'{mode}-seed{seed}.wav'; stem=out/f'{mode}-seed{seed}_words.csv'
        align(SimpleNamespace(audio=str(wav),transcript=str(target),out=str(stem),device=a.device,asr_model=a.asr_model,batch_size=a.batch_size,gap_threshold=a.gap_threshold))
        analyze(SimpleNamespace(target_text=str(target),asr_text=str(stem.with_suffix('.asr.txt')),reference_breaks=str(out/'original_words_breaks.csv'),generated_breaks=str(out/f'{mode}-seed{seed}_words_breaks.csv'),reviewed_breaks=review,original_audio=str(original),generated_audio=str(wav),tolerance=a.tolerance,outdir=str(out/f'report-seed{seed}')))

def main():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(required=True)
    x=sp.add_parser('prepare'); x.add_argument('--source',choices=['tedlium','ted_fallback'],required=True); x.add_argument('--tedlium-root'); x.add_argument('--workdir',default='/kaggle/working/f5-pilot/data'); x.set_defaults(func=prepare)
    x=sp.add_parser('generate'); x.add_argument('--reference-audio',required=True); x.add_argument('--reference-text',required=True); x.add_argument('--target-text',required=True); x.add_argument('--mode',choices=['short','long'],required=True); x.add_argument('--output-dir',default='/kaggle/working/f5-pilot/outputs'); x.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]); x.add_argument('--device',default='cuda'); x.set_defaults(func=generate)
    x=sp.add_parser('align'); x.add_argument('audio'); x.add_argument('--transcript',required=True); x.add_argument('--out',required=True); x.add_argument('--device',default='cuda'); x.add_argument('--asr-model',default='small.en'); x.add_argument('--batch-size',type=int,default=8); x.add_argument('--gap-threshold',type=float,default=.18); x.set_defaults(func=align)
    x=sp.add_parser('analyze'); x.add_argument('--target-text',required=True); x.add_argument('--asr-text',required=True); x.add_argument('--reference-breaks',required=True); x.add_argument('--generated-breaks',required=True); x.add_argument('--reviewed-breaks',required=True); x.add_argument('--original-audio'); x.add_argument('--generated-audio'); x.add_argument('--tolerance',type=float,default=.12); x.add_argument('--outdir',required=True); x.set_defaults(func=analyze)
    x=sp.add_parser('run'); x.add_argument('--source',choices=['tedlium','ted_fallback'],required=True); x.add_argument('--tedlium-root'); x.add_argument('--mode',choices=['short','long'],default='short'); x.add_argument('--workdir',default='/kaggle/working/f5-pilot/data'); x.add_argument('--output-dir',default='/kaggle/working/f5-pilot/outputs'); x.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]); x.add_argument('--device',default='cuda'); x.add_argument('--asr-model',default='small.en'); x.add_argument('--batch-size',type=int,default=8); x.add_argument('--gap-threshold',type=float,default=.18); x.add_argument('--tolerance',type=float,default=.12); x.add_argument('--reviewed-breaks'); x.set_defaults(func=full_run)
    a=p.parse_args(); a.func(a)
if __name__=='__main__': main()
