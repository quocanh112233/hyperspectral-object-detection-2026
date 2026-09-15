"""Tai mot mau anh PNG ve de EDA pho (moi anh ~4MB). Mac dinh 200 anh train."""
import os, sys, csv, random
from concurrent.futures import ThreadPoolExecutor
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "hyperspectral-object-detection-challenge-2026"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "train"
PREFIX = {"train": "data_train/data_train/VIS/", "test": "data_test/data_test/VIS/"}[SPLIT]
DST = os.path.join(ROOT, "data", "sample", SPLIT)

os.makedirs(DST, exist_ok=True)
with open(os.path.join(ROOT, "data", "raw", "file_list.csv")) as f:
    names = [r["name"] for r in csv.DictReader(f)
             if r["name"].startswith(PREFIX) and r["name"].endswith(".png")]
random.seed(0)
pick = random.sample(names, min(N, len(names)))
todo = [n for n in pick if not os.path.exists(os.path.join(DST, os.path.basename(n)))]
print(f"chon {len(pick)} anh, can tai {len(todo)}", flush=True)

api = KaggleApi(); api.authenticate()
done = [0]

def grab(name):
    for attempt in range(3):
        try:
            api.competition_download_file(COMP, name, path=DST, quiet=True)
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"  {done[0]}/{len(todo)}", flush=True)
            return None
        except Exception as e:
            if attempt == 2:
                return (name, repr(e))

with ThreadPoolExecutor(max_workers=8) as ex:
    fails = [r for r in ex.map(grab, todo) if r]
print(f"XONG: {done[0]} thanh cong, {len(fails)} that bai", flush=True)
for n, e in fails[:5]:
    print("  LOI", n, e)
