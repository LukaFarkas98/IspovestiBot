import requests
from bs4 import BeautifulSoup
import json
import hashlib
from datetime import datetime, timedelta
import time
import os

BASE_URL = "https://ispovesti.com/sort/calendar"
HEADERS = {"User-Agent": "Mozilla/5.0 (learning project)"}

TARGET_CONFESSIONS = 100_000
OUTPUT_FILE = "confessions_Ispovesti_Archive_raw.json"

CONFESSION_CLASS = "confession"
TEXT_CLASS = "confession-text"
VALUE_CLASS = "confession-value"
TIMESTAMP_CLASS = "confession-timestamp"

APPROVE_PREFIX = "approve-count-"
DISAPPROVE_PREFIX = "disapprove-count-"

# --------------------------
# Buffered saving config
# --------------------------
SAVE_BUFFER_SIZE = 50
save_buffer = []

session = requests.Session()
session.headers.update(HEADERS)

def hash_text(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def sendReq(url):
    MAX_RETRIES = 5

    response = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=20)
            response.encoding = "utf-8"
            return response
        except requests.exceptions.ReadTimeout:
            print(f"    ⚠ Timeout (attempt {attempt+1}/{MAX_RETRIES}), retrying...")
            time.sleep(2 + attempt * 2)
        except requests.exceptions.RequestException as e:
            print(f"    ❌ Request error: {e}")
            return None
    print("    ⛔ Too many retries, skipping page")
    return None 

# --------------------------
# Load progress if exists
# --------------------------
def load_progress():
    seen_hashes = set()
    all_confessions = []

    if os.path.exists("confessions_Archive_100k.jsonl"):
        print("📂 Found existing JSONL, loading progress...")
        with open("confessions_Archive_100k.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                all_confessions.append(item)
                seen_hashes.add(hash_text(item["text"]))

        if all_confessions:
            # Continue from the last date scraped
            last_date_str = all_confessions[-1]["date"]
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
            resume_date = last_date
            print(f"🔄 Resuming from {resume_date.strftime('%d/%m/%Y')}")
            return all_confessions, seen_hashes, resume_date

    # No existing data
    return [], set(), datetime(2025, 12, 17)

# Initialize
all_confessions, seen_hashes, current_date = load_progress()
min_date = datetime(2010, 1, 1)  # safety cutoff

def scrape():
    global all_confessions, seen_hashes, current_date
    while len(all_confessions) < TARGET_CONFESSIONS and current_date >= min_date:
        day = current_date.strftime("%d")
        month = current_date.strftime("%m")
        year = current_date.strftime("%Y")

        page = 1
        print(f"\n📅 Scraping date {day}/{month}/{year}")

        while True:
            url = f"{BASE_URL}/{day}/{month}/{year}//{page}"
            response = sendReq(url)

            if response is None:
                print("  ⛔ Skipping page due to request failure")
                break

            soup = BeautifulSoup(response.text, "lxml")
            confs = soup.find_all("div", class_=CONFESSION_CLASS)

            if not confs:
                print(f"  ⛔ No more pages for this date (page {page})")
                break

            print(f"  📄 Page {page} — {len(confs)} confessions")

            for conf in confs:
                text_div = conf.find("div", class_=TEXT_CLASS)
                if not text_div:
                    continue

                text = text_div.get_text(strip=True)
                if len(text) < 40:
                    continue

                h = hash_text(text)
                if h in seen_hashes:
                    continue

                seen_hashes.add(h)

                approve = conf.find(
                    "div",
                    class_=VALUE_CLASS,
                    id=lambda x: x and x.startswith(APPROVE_PREFIX)
                )
                disapprove = conf.find(
                    "div",
                    class_=VALUE_CLASS,
                    id=lambda x: x and x.startswith(DISAPPROVE_PREFIX)
                )
                timestamp = conf.find("div", class_=TIMESTAMP_CLASS)

                item = {
                    "text": text,
                    "approve": approve.get_text(strip=True) if approve else None,
                    "disapprove": disapprove.get_text(strip=True) if disapprove else None,
                    "timestamp": timestamp.get_text(strip=True) if timestamp else None,
                    "date": f"{year}-{month}-{day}"
                }

                # --------------------------
                # Buffered save logic
                # --------------------------
                save_buffer.append(item)
                if len(save_buffer) >= SAVE_BUFFER_SIZE:
                    flush_buffer()

                all_confessions.append(item)

                if len(all_confessions) >= TARGET_CONFESSIONS:
                    print("🎯 Target reached!")
                    flush_buffer()
                    return all_confessions
                
            time.sleep(0.7)   # between pages    
            page += 1

        current_date -= timedelta(days=1)
        time.sleep(1.2)   # between dates

    flush_buffer()
    return all_confessions

# --------------------------
# Save functions
# --------------------------
def save(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_one(item):
    """This is now optional, using buffer instead."""
    with open("confessions_Archive_100k.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def flush_buffer():
    global save_buffer
    if not save_buffer:
        return
    with open("confessions_Archive_100k.jsonl", "a", encoding="utf-8") as f:
        for item in save_buffer:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    save_buffer = []

# --------------------------
# Run scraper
# --------------------------
data = scrape()
save(data)
print(f"\n✅ Saved {len(data)} confessions")
