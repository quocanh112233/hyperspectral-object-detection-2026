import torch, subprocess, socket
print("torch:", torch.__version__)
print("cuda kha dung:", torch.cuda.is_available())
print("so GPU:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name}  {p.total_memory/1e9:.1f} GB")
try:
    socket.gethostbyname("pypi.org"); print("internet: CO")
except Exception as e:
    print("internet: KHONG -", e)
