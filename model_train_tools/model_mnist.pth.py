import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os


# === 1. 定义改进后的 CNN (BetterCNN) ===
class BetterCNN(nn.Module):
    def __init__(self):
        super(BetterCNN, self).__init__()

        # 第一层卷积块: Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 新增: 归一化，提升抗干扰能力

        # 第二层卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 新增

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)  # 新增: 丢弃50%神经元，防止死记硬背
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # FC Block
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc2(x)
        return x


# === 2. 训练配置 ===
def train_optimized():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用设备: {device}")

    # --- 策略A: 数据增强 (Data Augmentation) ---
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),  # 随机旋转 -15~15度 (模拟写歪了)
        transforms.RandomAffine(  # 随机仿射变换
            degrees=0,
            translate=(0.1, 0.1),  # 上下左右平移 10% (模拟没写在正中间)
            scale=(0.9, 1.1)  # 大小缩放 0.9~1.1倍
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 测试集不需要增强，只需要归一化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据
    train_dataset = datasets.MNIST('./data_cache', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 初始化模型
    model = BetterCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # --- 策略B: 增加训练轮数 (Epochs) ---
    epochs = 5
    print(f"🚀 开始训练 (计划 {epochs} 轮)...")

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        # 打印每一轮的成绩
        acc = 100. * correct / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs} | 平均Loss: {total_loss / len(train_loader):.4f} | 训练集准确率: {acc:.2f}%")

    # 保存模型
    save_path = "../models/model_a.pth"
    # 确保目录存在
    if not os.path.exists("../models"):
        os.makedirs("../models")

    torch.save(model.state_dict(), save_path)
    print(f"✅ 优化后的模型已保存: {save_path}")


if __name__ == '__main__':
    train_optimized()