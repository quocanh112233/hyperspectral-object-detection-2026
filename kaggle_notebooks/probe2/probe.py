import os, glob, hashlib
for root, dirs, files in os.walk("/kaggle/input"):
    if root.count("/") - 2 <= 4:
        print(root, "| dirs:", dirs[:5], "| files:", len(files), files[:3], flush=True)
print()
for pat, label in [("/kaggle/input/**/data_train/**/VIS/*.png", "anh train"),
                   ("/kaggle/input/**/Annotations/**/*.xml", "nhan train"),
                   ("/kaggle/input/**/data_test/**/VIS/*.png", "anh test")]:
    fs = glob.glob(pat, recursive=True)
    print(f"{label}: {len(fs)}")
    if fs:
        f = sorted(fs)[0]
        print("   vi du:", f, os.path.getsize(f), "bytes")
        print("   md5:", hashlib.md5(open(f,'rb').read()).hexdigest())
