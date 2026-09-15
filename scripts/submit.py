"""Nop file len Kaggle, co va loi treo cua thu vien kaggle.

LOI GOC: KaggleApi.upload_complete boc file trong TqdmBufferedReader (chi de hien
thanh tien trinh). requests khong xac dinh duoc do dai -> dung Transfer-Encoding:
chunked. Endpoint resumable upload cua GCS khong chap nhan chunked nen treo vo han
(file 341 byte cung treo, khong lien quan bang thong).

Dung:  python scripts/submit.py <file.csv> "ghi chu"
"""
import os, sys, requests
from kaggle.api.kaggle_api_extended import KaggleApi, ResumableUploadResult

COMP = "hyperspectral-object-detection-challenge-2026"


def upload_complete(self, path, url, quiet, resume=False):
    size = os.path.getsize(path)
    with open(path, "rb") as fp:
        r = requests.put(url, data=fp,
                         headers={"Content-Length": str(size)},
                         timeout=(30, 900))
    if not quiet:
        print(f"  upload: HTTP {r.status_code} ({size/1e6:.2f} MB)")
    if self._is_upload_successful(r):
        return ResumableUploadResult.COMPLETE
    print("  loi upload:", r.text[:300])
    return ResumableUploadResult.FAILED


KaggleApi.upload_complete = upload_complete

if __name__ == "__main__":
    path = sys.argv[1]
    msg = sys.argv[2] if len(sys.argv) > 2 else "submission"
    assert os.path.exists(path), f"khong thay {path}"
    api = KaggleApi(); api.authenticate()
    print(f"nop {path} ({os.path.getsize(path)/1e6:.2f} MB)")
    print(api.competition_submit(path, msg, COMP, quiet=False))
    print("\n=== cac lan nop ===")
    api.competition_submissions_cli(COMP, csv_display=True)
