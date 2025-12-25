import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("❌ 警告：未检测到 GPU，正在使用 CPU！")