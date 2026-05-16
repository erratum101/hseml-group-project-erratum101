import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fix_baseline():
    p = ROOT / "notebooks" / "2_baseline.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    if len(nb["cells"]) > 1 and nb["cells"][0]["cell_type"] == nb["cells"][1]["cell_type"] == "markdown":
        if nb["cells"][0]["source"] == nb["cells"][1]["source"]:
            del nb["cells"][1]
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "df_model[features]" in src:
            cell["source"] = [src.replace("df_model", "df")]
        if "dropna(subset=features" in src and "df_model =" in src:
            cell["source"] = [src.replace("df_model", "df")]
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def fix_experiments():
    p = ROOT / "notebooks" / "3_experements.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    seen = set()
    new_cells = []
    for cell in nb["cells"]:
        key = (cell["cell_type"], "".join(cell.get("source", []))[:120])
        if key in seen:
            continue
        seen.add(key)
        new_cells.append(cell)
    nb["cells"] = new_cells
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    fix_baseline()
    fix_experiments()
    print("Fixed duplicate cells.")
