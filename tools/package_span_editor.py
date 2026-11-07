"""Create a small DataHub overlay zip with the current span-editor code.

Run from any directory: python tools/package_span_editor.py --output PATH.zip
The archive paths are relative to the RevProof project root.
"""

import argparse
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "README.md", "fruit_data.py", "fruit_local.py", "fruit_span.py",
    "evaluate_fruit_gate.py", "test_span_editor.py", "requirements-datahub.txt",
    "research/README.md", "research/datahub_span_editor.md",
    "results/README.md",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    files = [ROOT / name for name in FILES] + sorted((ROOT / "span_editor").glob("*.py"))
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package files: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            body = path.read_bytes()
            archive.writestr(name, body)
            manifest.append(f"{hashlib.sha256(body).hexdigest()}  {name}")
        archive.writestr("SPAN_EDITOR_MANIFEST.sha256", "\n".join(manifest) + "\n")
    print(f"{args.output.resolve()} ({len(files)} files, {args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
