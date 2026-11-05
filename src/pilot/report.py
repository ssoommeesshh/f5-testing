from pathlib import Path
import json

def write_html(summary: dict, out: Path, original_audio: str | None = None, generated_audio: str | None = None) -> None:
    rows = "".join(f"<tr><th>{k}</th><td><pre>{json.dumps(v, indent=2)}</pre></td></tr>" for k,v in summary.items())
    players = "".join(f'<h2>{name}</h2><audio controls src="{path}"></audio>' for name,path in [('Original',original_audio),('Generated',generated_audio)] if path)
    out.write_text(f"<!doctype html><title>F5 timing pilot</title><h1>F5 timing pilot</h1><p>Relative to this speaker's recording; automated measurements are not delivery judgments.</p>{players}<table>{rows}</table>", encoding="utf-8")
