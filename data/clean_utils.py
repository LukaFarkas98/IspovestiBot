import re
import unicodedata
import json


INPUT_FILE = "../Clustering/confessions_COMBINED_140K.jsonl"
OUTPUT_FILE = "confessions_Archive_140k_clean.jsonl"

PREFIX = "Ispovest:\n"


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Unicode normalize (fix weird chars)
    text = unicodedata.normalize("NFKC", text)

    # Remove non-breaking spaces & control chars
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


bad_lines = 0
kept = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line_num, line in enumerate(fin, start=1):
        line = line.strip()

        # skip empty lines
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            bad_lines += 1
            print(f"[WARN] Bad JSON on line {line_num}: {e}")
            continue

        if "text" not in item:
            bad_lines += 1
            print(f"[WARN] Missing 'text' on line {line_num}")
            continue

        text = clean_text(item["text"])
        if not text:
            continue

        fout.write(
            json.dumps(
                {"text": PREFIX + text},
                ensure_ascii=False
            ) + "\n"
        )
        kept += 1


print(f"✅ Cleaning done → {OUTPUT_FILE}")
print(f"🧹 Kept {kept} entries")
print(f"⚠️ Skipped {bad_lines} bad lines")
