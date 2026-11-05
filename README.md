# F5-TTS real-presentation timing pilot

This is an evaluation pipeline under development, not measured evidence about F5-TTS. It measures timing **relative to one speaker's recording**; a recorded break is not an objective delivery requirement. No real TED/F5 inference was performed while building this repository.

## Source decision

The intended primary is TED-LIUM Release 3 legacy-test `DanBarber_2010` (SPHERE audio plus segmented timestamped STM). The complete source record, expected paths, attribution, archive checksum, and selected windows are in [the manifest](manifests/dan_barber_2010.json). On 2026-11-05 its official full-corpus URL returned HTTP 404; no reliable talk-level TED-LIUM download was found. The code does not download the corpus.

An explicit `ted_fallback` option downloads the same talk from the official TED page with English captions. Its runtime manifest labels it **not TED-LIUM**. It is never a silent substitution and no media or weights are committed.

## Kaggle GPU quick start

Enable **Internet** and **GPU**, then run this one command after installation:

```bash
!git clone https://github.com/ssoommeesshh/f5-testing.git
%cd f5-testing
!pip install -r requirements.txt
!pip install -e .
!pytest -q
!python -m pilot.cli run --source ted_fallback --mode short --workdir /kaggle/working/f5-pilot/data --output-dir /kaggle/working/f5-pilot/outputs --seeds 11
```

The command stops on every failed dependency or source artifact. `prepare` fetches media/captions, validates an 8-second non-overlapping reference window, and writes `ref.wav`, `ref.txt`, `target-short.txt`, `target-long.txt`, `original-target-short.wav`, `original-target-long.wav`, and `original-target.wav`. Empty caption windows fail rather than producing invented text.

For the full three-seed target run:

```bash
!python -m pilot.cli run --source ted_fallback --mode long --workdir /kaggle/working/f5-pilot/data --output-dir /kaggle/working/f5-pilot/outputs --seeds 11 22 33
!tar -czf /kaggle/working/f5-pilot-export.tgz /kaggle/working/f5-pilot
```

For a lawfully mounted TED-LIUM release, replace source options with `--source tedlium --tedlium-root /kaggle/input/tedlium3`.

## What the command does

Generation uses the official installed `f5-tts` Python package and `F5TTS_v1_Base`. Upstream `f5-tts_infer-cli` has no `--seed` flag (verified against upstream source on 2026-11-05), so `pilot.f5_runner` seeds Python, NumPy, and Torch before invoking the same package inference utilities. Each output has a saved seed/config sidecar. The baseline passes text unchanged and never uses silence removal or inserted audio gaps.

The long target is labelled a multi-sentence pilot, **not** a single long model generation. The current F5 wrapper does not expose automatic internal chunks; it records an empty boundary list rather than fabricating one.

`align` first force-aligns the **known target transcript** to audio with WhisperX's English alignment model. It separately runs ASR and saves `*.asr.txt`/`*.asr.json`; ASR never supplies timing words. Failed/non-monotonic known-text alignment yields no score. `analyze` automatically consumes that saved ASR output and writes `summary.json`, `per_break.csv`, and an HTML report with original/generated audio players.

Only rows manually marked `break` in [annotations/reviewed_breaks.csv](annotations/reviewed_breaks.csv) are reference break labels. Keep human decisions (problematic / acceptable alternative / alignment or ASR error) separate from automated output.

## Current limitations

The HTML report is intentionally compact: it has audio players and summary/per-break data, but not yet interactive word-boundary/chunk plots or ASR-error timestamp visualizations. Treat these as planned enhancements, not completed features. Review low-confidence/failed alignment and listen before publishing any interpretation.

## Validation

CPU tests cover VTT/STM extraction, source-plan validation, transcript normalization/WER, gap accounting, and failed alignment rejection. They contain no downloaded speech, model weights, or real F5 inference.

## References

- [Official F5-TTS](https://github.com/SWivid/F5-TTS)
- [TED-LIUM 3](https://arxiv.org/abs/1805.04699)
- [WhisperX](https://github.com/m-bain/whisperX)
