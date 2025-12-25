# generate_mnist_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# 1. 定义一个超简单的 CNN (LeNet变体)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个数字
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28->14
        x = self.pool(self.relu(self.conv2(x)))  # 14->7
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 2. 训练一轮
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data_cache', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("🚀 正在快速训练 MNIST 模型 (约需 30秒)...")
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"训练进度: {batch_idx}/{len(loader)}")

    torch.save(model.state_dict(), "../models/model_a.pth")
    print("✅ MNIST 模型已保存为 models/model_a.pth")


if __name__ == '__main__':
    import os

    os.makedirs("../models", exist_ok=True)
    train()