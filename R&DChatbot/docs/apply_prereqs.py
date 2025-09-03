import json, shutil
from pathlib import Path

# Folder containing catalog_master.jsonl and the per-major JSONLs
DOCS_DIR = Path("docs")

# Prerequisite map (normalized to UPPER codes)
PREREQ = {
    "COMP503": {"prerequisites": ["COMP500", "ENSE501"],                       "prereq_text": "COMP500/ENSE501"},
    "COMP602": {"prerequisites": ["COMP603", "COMP610"],                       "prereq_text": "COMP603 or COMP610"},
    "COMP603": {"prerequisites": ["COMP503", "COMP610", "ENSE502"],            "prereq_text": "COMP503/COMP610/ENSE502"},
    "COMP604": {"prerequisites": ["COMP503", "ENSE502", "ENSE504", "COMP504"], "prereq_text": "[COMP503/ENSE502/ENSE504] or COMP504"},
    "COMP607": {"prerequisites": ["COMP501"],                                  "prereq_text": "COMP501"},
    "COMP609": {"prerequisites": ["COMP500", "COMP504"],                       "prereq_text": "COMP500 and COMP504"},
    "COMP610": {"prerequisites": ["COMP503", "ENSE502", "ENSE602"],            "prereq_text": "COMP503/ENSE502/ENSE602"},
    "COMP611": {"prerequisites": ["COMP610"],                                  "prereq_text": "COMP610"},
    "COMP612": {"prerequisites": ["MATH503", "MATH502", "COMP603", "COMP610"], "prereq_text": "[MATH503 or MATH502] and [COMP603 or COMP610]"},
    "COMP613": {"prerequisites": ["COMP500", "MATH502", "MATH503"],            "prereq_text": "COMP500 and [MATH502 or MATH503]"},
    "COMP615": {"prerequisites": ["COMP517"],                                  "prereq_text": "COMP517"},
    "COMP616": {"prerequisites": ["MATH502", "MATH503"],                       "prereq_text": "MATH502 or MATH503"},
    "ENEL611": {"prerequisites": ["COMP504", "ENEL504"],                       "prereq_text": "COMP504 or ENEL504"},
    "STAT603": {"prerequisites": ["MATH502", "MATH503"],                       "prereq_text": "MATH502 or MATH503"},
    "COMP701": {"prerequisites": ["COMP500"],                                  "prereq_text": "COMP500"},
    "COMP711": {"prerequisites": ["COMP610", "COMP613"],                       "prereq_text": "COMP610 or COMP613"},
    "COMP712": {"prerequisites": ["COMP603", "ENSE502"],                       "prereq_text": "COMP603/ENSE502"},
    "COMP713": {"prerequisites": ["COMP611"],                                  "prereq_text": "COMP611"},
    "COMP714": {"prerequisites": ["COMP609"],                                  "prereq_text": "COMP609"},
    "COMP715": {"prerequisites": ["ENEL611"],                                  "prereq_text": "ENEL611"},
    "COMP716": {"prerequisites": ["COMP611", "ENGE501", "COMP610"],            "prereq_text": "COMP611 or [ENGE501 and COMP610]"},
    "COMP717": {"prerequisites": ["COMP500"],                                  "prereq_text": "COMP500 or equivalent; 60 points at level 6 major"},
    "COMP721": {"prerequisites": ["COMP603", "ENSE600"],                       "prereq_text": "COMP603/ENSE600"},
    "COMP729": {"prerequisites": ["COMP504", "ENEL504"],                       "prereq_text": "COMP504/ENEL504"},
    "ENSE701": {"prerequisites": ["COMP603", "COMP610", "ENSE600"],            "prereq_text": "COMP603 or [COMP610/ENSE600]"},
}

def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def write_jsonl(path: Path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def apply_prereqs(rec: dict):
    code = (rec.get("code") or "").upper()
    if not code or code not in PREREQ:
        return False
    patch = PREREQ[code]
    rec["prerequisites"] = patch["prerequisites"]
    rec["prereq_text"] = patch["prereq_text"]
    return True

def main():
    if not DOCS_DIR.exists():
        raise SystemExit(f"docs/ not found: {DOCS_DIR.resolve()}")

    jsonl_files = sorted(DOCS_DIR.glob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit("No .jsonl files found in docs/")

    for jf in jsonl_files:
        records = load_jsonl(jf)
        changed = 0

        # backup once per file
        bak = jf.with_suffix(jf.suffix + ".bak")
        shutil.copyfile(jf, bak)

        for r in records:
            # Normalize fields the app expects
            r["code"] = (r.get("code") or "").upper()
            if "semesters" not in r:
                r["semesters"] = r.get("semester", []) or []
            if "prerequisites" not in r:
                r["prerequisites"] = r.get("prereqs", []) or []

            if apply_prereqs(r):
                changed += 1

        write_jsonl(jf, records)
        print(f"✔ {jf.name}: updated prerequisites for {changed} course(s) — backup at {bak.name}")

if __name__ == "__main__":
    main()
