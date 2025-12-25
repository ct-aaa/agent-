import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import time

# --- 配置区域 ---
DATASET_PATH = '../datasets/dataset_C'
SAVE_MODEL_PATH = '../models/model_c.pth'
SAVE_TXT_PATH = '../models/model_c_classes.txt'
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


def train_enhanced():
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 MobileNetV3 训练模式 (Device: {device})")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ 错误: 找不到数据集 {DATASET_PATH}")
        return

    # ==================================================
    # 2. 增强策略
    # ==================================================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ==================================================
    # 3. 加载与划分
    # ==================================================
    full_dataset = datasets.ImageFolder(root=DATASET_PATH)
    classes = full_dataset.classes
    num_classes = len(classes)

    print(f"📂 数据集包含 {len(full_dataset)} 张图片, {num_classes} 个类别")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform: x = self.transform(x)
            return x, y

        def __len__(self): return len(self.subset)

    train_data = TransformedDataset(train_subset, transform=train_transform)
    val_data = TransformedDataset(val_subset, transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ==================================================
    # 4. 模型初始化 (MobileNet V3 Small)
    # ==================================================
    print("🏗️  初始化 MobileNet V3 Small...")

    model = models.mobilenet_v3_small(weights='DEFAULT')

    # --- 冻结骨干网络 ---
    for param in model.parameters():
        param.requires_grad = False

    # --- 修改分类头 ---
    # 获取分类器输入层特征数
    num_ftrs = model.classifier[3].in_features

    # 替换最后一层
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, num_classes)
    )

    model = model.to(device)

    # ==================================================
    # 5. 优化器与损失 (已修正错误)
    # ==================================================
    criterion = nn.CrossEntropyLoss()

    # ✅ 修正点：这里必须优化 model.classifier 的参数，而不是 model.fc
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 学习率调整
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # ==================================================
    # 6. 训练循环
    # ==================================================
    best_acc = 0.0
    print(f"\n🔥 开始训练 (Epochs: {EPOCHS})...")

    for epoch in range(EPOCHS):
        start_time = time.time()

        # Train
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / train_size
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = val_correct / val_total
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} [{elapsed:.0f}s] Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.1%} | Val Acc: {val_acc:.1%}",
            end="")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            with open(SAVE_TXT_PATH, 'w', encoding='utf-8') as f:
                f.write('\n'.join(classes))
            print(" ⭐ Saved Best")
        else:
            print("")

    print(f"\n✅ 训练完成。最佳验证准确率: {best_acc:.1%}")
    print(f"模型已保存: {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    train_enhanced()