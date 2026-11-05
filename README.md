# F5-TTS real-presentation timing pilot

This repository is a cautious Kaggle GPU evaluation pilot for F5-TTS on one real, single-speaker English TED presentation. It measures timing **relative to that speaker's recording**. It does not claim that F5-TTS has poor delivery, that a recorded pause is always mandatory, or that a forced alignment proves the words were spoken correctly.

## Source status and choice

The intended primary source is the TED-LIUM Release 3 legacy test talk `DanBarber_2010` (“How I fell in love with a fish”), SPHERE audio plus its timestamped STM test transcript. The full, auditable source record is [manifests/dan_barber_2010.json](manifests/dan_barber_2010.json). On 2026-11-05, the official URL `https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz` returned HTTP 404. It is also a full-corpus archive, not a small per-talk route. Therefore this repository neither downloads nor redistributes it.

`prepare --source tedlium` accepts an already legitimately obtained release and verifies the expected paths. Only if a user explicitly selects `--source ted_fallback` does it use the official TED page for the same talk and fetch its audio and English captions at runtime via `yt-dlp`. That fallback is plainly recorded in its runtime manifest and is **not TED-LIUM**. No audiobook or meeting is substituted. TED/LIUM attribution and reuse notes are in the manifest; do not redistribute the fetched media without confirming terms.

The 8-second reference window (60–68 s), 25-second quick target (180–205 s), and 60-second long target (180–240 s) are provisional. `prepare` must validate their transcript coverage and non-overlap before a run is considered valid. Exact extracted text/windows go in `/kaggle/working`, not Git.

## Kaggle: exact runbook

1. Create a notebook, turn on **GPU** and **Internet**, then run:

```bash
!git clone https://github.com/ssoommeesshh/f5-testing.git
%cd f5-testing
!pip install -r requirements.txt
!pip install -e .
!pytest -q
```

2. Fetch source data (the fallback must be acknowledged in your experiment notes):

```bash
!python -m pilot.cli prepare --source ted_fallback --workdir /kaggle/working/f5-pilot/data
# Or, for a locally mounted, lawfully obtained TED-LIUM release:
# !python -m pilot.cli prepare --source tedlium --tedlium-root /kaggle/input/tedlium3 --workdir /kaggle/working/f5-pilot/data
```

3. Make source clips/transcripts only after inspecting the VTT/STM timestamps and update `runtime_manifest.json`. Keep a clean 5–10 s same-speaker reference, exact reference transcript, and a non-overlapping target transcript. Do **not** change target wording, add commas/pause tokens, remove silence, or insert gaps. Then smoke-test one short generation:

```bash
!python -m pilot.cli generate --mode short --reference-audio /kaggle/working/f5-pilot/ref.wav --reference-text /kaggle/working/f5-pilot/ref.txt --target-text /kaggle/working/f5-pilot/target-short.txt --output-dir /kaggle/working/f5-pilot/outputs --seeds 11
```

4. Full pilot uses three fixed seeds and explicitly multi-chunk long text:

```bash
!python -m pilot.cli generate --mode long --reference-audio /kaggle/working/f5-pilot/ref.wav --reference-text /kaggle/working/f5-pilot/ref.txt --target-text /kaggle/working/f5-pilot/target-long.txt --output-dir /kaggle/working/f5-pilot/outputs --seeds 11 22 33
!python -m pilot.cli align /kaggle/working/f5-pilot/original-target.wav --out /kaggle/working/f5-pilot/align/original_words.csv
!python -m pilot.cli align /kaggle/working/f5-pilot/outputs/<generated>.wav --out /kaggle/working/f5-pilot/align/generated_words.csv
!python -m pilot.cli analyze --reference-text /kaggle/working/f5-pilot/target-long.txt --hypothesis-text /kaggle/working/f5-pilot/asr.txt --outdir /kaggle/working/f5-pilot/report
!tar -czf /kaggle/working/f5-pilot-export.tgz /kaggle/working/f5-pilot
```

WhisperX (English `small.en` plus English alignment model) independently aligns the known target to the original and every generated WAV. It writes word times and candidate gaps; its separate ASR transcription needs to be retained per output before `analyze`. If timing is absent, non-monotonic, or low-confidence, stop and inspect audio—do not report a pause score.

## Review protocol and outputs

`align` writes `*_words.csv`, `candidates.csv`, and raw aligned segments. Review candidates into a copy of `annotations/reviewed_breaks.csv`; only rows marked `break` are reference labels. Record each highlighted difference as `perceptually_problematic`, `acceptable_alternative_phrasing`, or `alignment_or_asr_error` in a reviewer CSV. Automated results and human judgments must stay separate.

`analyze` writes `summary.json` and a compact HTML report. Before publication, augment it with per-break match/missing/extra rows using a stated tolerance, ASR substitution/deletion/insertion timestamps, audio players, and plots of word/gap/F5 chunk boundaries. The current code deliberately does not fabricate any of those missing runtime artifacts.

## Validation status

Unit tests cover normalization/WER, word/gap accounting, and failed alignment rejection. They are local, CPU-only checks. No TED media download, F5 checkpoint fetch, GPU inference, alignment, ASR, or pause result was run during repository construction.

## References

- Official F5-TTS: https://github.com/SWivid/F5-TTS (uses `F5TTS_v1_Base` and caches checkpoint downloads at runtime).
- Hernandez et al., *TED-LIUM 3* (2018): https://arxiv.org/abs/1805.04699.
- WhisperX: https://github.com/m-bain/whisperX.
