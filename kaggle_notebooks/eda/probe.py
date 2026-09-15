import os
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count("/") - 2
    if depth <= 4:
        print(root, "| dirs:", dirs[:6], "| files:", len(files), files[:4], flush=True)
