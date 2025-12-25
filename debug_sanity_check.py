import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# --- 配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10  # 小批量，方便观察
LR = 0.001


def debug_training():
    print(f"🚀 开始进行代码健全性检查 (Sanity Check)...")

    # 1. 准备数据 (会自动下载全量，但我们只用一点点)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 下载全量数据
    full_dataset = datasets.CIFAR10(root='./data_cache', train=True, download=True, transform=transform)

    # 【关键步骤】只取前 100 张图片！
    # 使用 Subset 创建一个迷你数据集
    indices = list(range(100))
    mini_dataset = Subset(full_dataset, indices)

    # 放入 DataLoader
    train_loader = DataLoader(mini_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"📊 全量数据: {len(full_dataset)} 张 -> 测试数据: {len(mini_dataset)} 张")

    # 2. 定义模型 (ResNet18)
    model = models.resnet18(weights=None)  # 从头开始练
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    # 3. 优化器与 Loss
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 4. 疯狂训练 50 轮 (目标是 Loss -> 0)
    model.train()
    for epoch in range(50):
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        # 只打印关键轮次
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02d} | Loss: {total_loss:.4f} | Acc: {acc:.2f}%")

        # 如果准确率达到 100%，说明代码没问题
        if acc == 100.0:
            print(f"\n✅ 成功！模型在 Epoch {epoch + 1} 完美拟合了小样本。")
            print("结论：你的训练代码逻辑是正确的，可以放心去跑全量数据了。")
            return

    print("\n❌ 警告：50轮后仍未拟合小样本，请检查代码 (学习率是否太大？模型是否太简单？)")


if __name__ == "__main__":
    debug_training()