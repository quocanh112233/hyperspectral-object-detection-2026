"""Tai toan bo 3000 nhan VOC XML (~5MB) ve data/annotations/. Chay da luong vi API cham ~2s/file."""
import os, sys, csv, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "hyperspectral-object-detection-challenge-2026"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "data", "annotations")
PREFIX = "data_train/data_train/Annotations/VIS/"

os.makedirs(DST, exist_ok=True)
with open(os.path.join(ROOT, "data", "raw", "file_list.csv")) as f:
    names = [r["name"] for r in csv.DictReader(f)
             if r["name"].startswith(PREFIX) and r["name"].endswith(".xml")]
todo = [n for n in names if not os.path.exists(os.path.join(DST, os.path.basename(n)))]
print(f"tong {len(names)} nhan, can tai {len(todo)}", flush=True)

api = KaggleApi(); api.authenticate()
done = [0]

def grab(name):
    for attempt in range(3):
        try:
            api.competition_download_file(COMP, name, path=DST, quiet=True)
            done[0] += 1
            if done[0] % 200 == 0:
                print(f"  {done[0]}/{len(todo)}", flush=True)
            return None
        except Exception as e:
            if attempt == 2:
                return (name, repr(e))
            time.sleep(2 * (attempt + 1))

with ThreadPoolExecutor(max_workers=4) as ex:
    fails = [r for r in ex.map(grab, todo) if r]

print(f"XONG: {done[0]} thanh cong, {len(fails)} that bai", flush=True)
for n, e in fails[:10]:
    print("  LOI", n, e)
sys.exit(1 if fails else 0)
