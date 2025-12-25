import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import time

# --- 配置区域 ---
DATASET_PATH = 'datasets/dataset_C'
# 💡 读取你刚才那个已经有 60% 准确率的模型作为起点
LOAD_MODEL_PATH = 'models/model_c_finetuned.pth'
SAVE_MODEL_PATH = 'models/model_c_pro.pth'
SAVE_TXT_PATH = 'models/model_c_classes.txt'
BATCH_SIZE = 32
EPOCHS = 30
# 学习率再低一点，因为我们是在微调一个已经还不错的模型
LEARNING_RATE = 5e-5


def fine_tune_pro():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动终极优化模式 (Device: {device})")

    # ==================================================
    # 1. 强力数据增强 (关键改进点！)
    # ==================================================
    train_transform = transforms.Compose([
        # RandomResizedCrop 是核心：它会随机截取图片的一部分并放大
        # 这迫使模型学习局部特征，而不是死记硬背整张图
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),

        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # 增加旋转角度
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # 增加颜色干扰
        transforms.ToTensor(),

        # 随机擦除：随机把图片挖掉一块，强迫模型靠剩余部分识别
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据
    full_dataset = datasets.ImageFolder(root=DATASET_PATH)
    classes = full_dataset.classes
    num_classes = len(classes)

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

    train_loader = DataLoader(TransformedDataset(train_subset, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TransformedDataset(val_subset, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    # ==================================================
    # 2. 模型结构重建 (保持一致)
    # ==================================================
    print(f"🏗️  加载现有权重: {LOAD_MODEL_PATH}")
    model = models.mobilenet_v3_small(weights=None)

    # 重建分类头 (注意：这里我把 Dropout 加大到了 0.5)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.5),  # 💡 加大 Dropout，防止过拟合
        nn.Linear(num_ftrs, num_classes)
    )

    # 加载权重
    if os.path.exists(LOAD_MODEL_PATH):
        state_dict = torch.load(LOAD_MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("❌ 找不到上一轮的模型文件！")
        return

    # 全网解冻
    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)

    # ==================================================
    # 3. 损失函数改进 & 优化器
    # ==================================================
    # 💡 Label Smoothing: 标签平滑，防止模型盲目自信
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 💡 Weight Decay: 加大到 0.005，强力抑制过拟合
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.005)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)

    # ==================================================
    # 4. 训练循环
    # ==================================================
    best_acc = 0.0

    # 重新跑一遍验证集，看看起点在哪里
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

    start_acc = val_correct / val_total
    best_acc = start_acc
    print(f"🏁 当前基准准确率: {start_acc:.1%}")

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

        # 只有当验证集准确率没掉太多的时候，才更新学习率
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} [{elapsed:.0f}s] Loss: {epoch_loss:.4f} | Train: {train_acc:.1%} | Val: {val_acc:.1%}",
            end="")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            with open(SAVE_TXT_PATH, 'w', encoding='utf-8') as f:
                f.write('\n'.join(classes))
            print(" ⭐ New Best!")
        else:
            print("")

    print(f"\n✅ 优化完成。最佳验证准确率: {best_acc:.1%}")
    print(f"模型已保存: {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    fine_tune_pro()