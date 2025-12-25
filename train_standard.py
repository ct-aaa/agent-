import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# 配置

if not torch.cuda.is_available():
    print("⚠️ 警告: 当前正在使用 CPU 训练，速度会非常慢！请检查 CUDA 安装。")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
    print(f"✅ 成功调用 GPU: {torch.cuda.get_device_name(0)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)


def train_standard_model(dataset_name, save_path):
    print(f"\n{'=' * 10} 正在训练标准模型: {dataset_name} {'=' * 10}")

    # 1. 准备数据和网络
    if dataset_name == 'mnist':
        # MNIST 是灰度图(1通道)，需要转为3通道以适配 ResNet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_data = datasets.MNIST(root='./data_cache', train=True, download=True, transform=transform)
        classes = [str(i) for i in range(10)]
        model = models.resnet18(weights='DEFAULT')  # 使用预训练权重加速收敛
        model.fc = nn.Linear(model.fc.in_features, 10)

    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),  # 数据增强
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = datasets.CIFAR10(root='./data_cache', train=True, download=True, transform=transform)
        classes = train_data.classes  # ['airplane', 'automobile', 'bird', ...]
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, 10)

    # 2. 训练配置
    model = model.to(device)
    loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. 快速训练 (只跑 1-2 个 Epoch 足够应付作业)
    model.train()
    epochs = 1
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {i}, Loss: {loss.item():.4f}")

    # 4. 保存模型和类别表
    torch.save(model.state_dict(), save_path)

    # 保存 class.txt
    txt_path = save_path.replace('.pth', '_classes.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(classes))

    print(f"✅ 模型已保存: {save_path}")


if __name__ == "__main__":
    # 训练 MNIST -> model_a
    train_standard_model('mnist', 'models/model_a.pth')

    # 训练 CIFAR-10 -> model_b
    train_standard_model('cifar10', 'models/model_b.pth')

    print("\n所有标准模型训练完成！请手动处理 dataset_C。")