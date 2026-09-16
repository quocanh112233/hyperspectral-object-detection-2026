#!/usr/bin/env bash
# Do xem gioi han tai du lieu cuoc thi duoc go luc nao.
# Thu 30 phut mot lan bang file NHO NHAT (class.txt, 169 byte) de khong ton quota.
# Thoat ngay khi thanh cong.
cd /home/quocanh/quoc_anh/hyperspectral-object-detection-2026
source .venv/bin/activate
LOG=data/ratelimit_watch.log
COMP=hyperspectral-object-detection-challenge-2026
for i in $(seq 1 48); do
  T=$(date "+%Y-%m-%d %H:%M")
  OUT=$(kaggle competitions download $COMP -f class.txt \
        -p /tmp/claude-1000/-home-quocanh-quoc-anh/8862a4cf-f8d0-404d-8699-5b6371bc1782/scratchpad/rl \
        --force 2>&1 | tail -1)
  if echo "$OUT" | grep -q "429"; then
    echo "$T  van bi chan" >> $LOG
  else
    echo "$T  DA GO CHAN: $OUT" >> $LOG
    echo "DA GO CHAN luc $T"
    exit 0
  fi
  sleep 1800
done
echo "sau 24h van bi chan"
