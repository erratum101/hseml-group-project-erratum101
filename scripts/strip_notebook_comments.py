import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def strip_inline_comment(line: str) -> str:
    in_single = in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i].rstrip()
    return line


def clean_source(source: str) -> str:
    lines = []
    for line in source.splitlines():
        if line.strip().startswith("#"):
            continue
        cleaned = strip_inline_comment(line)
        if cleaned.strip():
            lines.append(cleaned)
    return "\n".join(lines) + ("\n" if lines else "")


def clean_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src = clean_source(src)
        if new_src:
            cell["source"] = [new_src]
        else:
            cell["source"] = []
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    for name in ["1_eda.ipynb", "2_baseline.ipynb", "3_experements.ipynb"]:
        clean_notebook(ROOT / "notebooks" / name)
        print("cleaned", name)
