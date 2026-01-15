import requests
from bs4 import BeautifulSoup
import json
import hashlib
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

BASE_URL = "https://ispovesti.com/sort/calendar"
HEADERS = {"User-Agent": "Mozilla/5.0 (learning project)"}

TARGET_CONFESSIONS = 200_000
OUTPUT_FILE = "confessions_Ispovesti_Archive_raw_200k.json"
JSONL_FILE = "confessions_Archive_200k.jsonl"

CONFESSION_CLASS = "confession"
TEXT_CLASS = "confession-text"
VALUE_CLASS = "confession-value"
TIMESTAMP_CLASS = "confession-timestamp"

APPROVE_PREFIX = "approve-count-"
DISAPPROVE_PREFIX = "disapprove-count-"

SAVE_BUFFER_SIZE = 50
save_buffer = []

session = requests.Session()
session.headers.update(HEADERS)

def hash_text(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

# --------------------------
# Resume logic (ADDED)
# --------------------------
def load_existing_data():
    seen = set()
    oldest_date = None

    if not os.path.exists(JSONL_FILE):
        return seen, None

    print("🔁 Resuming from existing JSONL...")

    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                text = item.get("text")
                date_str = item.get("date")

                if text:
                    seen.add(hash_text(text))

                if date_str:
                    d = datetime.strptime(date_str, "%Y-%m-%d")
                    if oldest_date is None or d < oldest_date:
                        oldest_date = d
            except Exception:
                continue

    if oldest_date:
        print(f"🧭 Oldest saved date: {oldest_date.strftime('%Y-%m-%d')}")

    return seen, oldest_date

def sendReq(url):
    MAX_RETRIES = 5
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

def parse_page(html, year, month, day):
    soup = BeautifulSoup(html, "lxml")
    confs = soup.find_all("div", class_=CONFESSION_CLASS)
    items = []
    for conf in confs:
        text_div = conf.find("div", class_=TEXT_CLASS)
        if not text_div:
            continue
        text = text_div.get_text(strip=True)

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
        items.append(item)
    return items

def flush_buffer():
    global save_buffer
    if not save_buffer:
        return
    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        for item in save_buffer:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    save_buffer = []

def scrape():
    all_confessions = []
    global seen_hashes

    seen_hashes, resume_date = load_existing_data()

    if resume_date:
        current_date = resume_date - timedelta(days=1)
    else:
        current_date = datetime(2015, 12, 17)

    min_date = datetime(2010, 1, 1)

    while len(seen_hashes) < TARGET_CONFESSIONS and current_date >= min_date:
        day = current_date.strftime("%d")
        month = current_date.strftime("%m")
        year = current_date.strftime("%Y")

        print(f"\n📅 Scraping date {day}/{month}/{year}")

        pages_to_scrape = [
            f"{BASE_URL}/{day}/{month}/{year}//{p}" for p in range(1, 6)
        ]

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(sendReq, url) for url in pages_to_scrape]
            for future in as_completed(futures):
                r = future.result()
                if r is None:
                    continue

                items = parse_page(r.text, year, month, day)
                for item in items:
                    save_buffer.append(item)
                    all_confessions.append(item)

                    if len(save_buffer) >= SAVE_BUFFER_SIZE:
                        flush_buffer()

                    if len(seen_hashes) >= TARGET_CONFESSIONS:
                        flush_buffer()
                        return all_confessions

        time.sleep(1.2)
        current_date -= timedelta(days=1)

    flush_buffer()
    return all_confessions

def save(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

data = scrape()
save(data)
print(f"\n✅ Saved {len(data)} confessions")
