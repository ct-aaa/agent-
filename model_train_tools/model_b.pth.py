import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# === 1. 配置 ===
DATASET_DIR = "../datasets/dataset_B"
MODEL_SAVE_PATH = "../models/model_b.pth"
CLASS_SAVE_PATH = "../models/model_b_classes.txt"
BATCH_SIZE = 32
EPOCHS = 10  # 稍微多训练几轮
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === 2. 自定义数据集读取器 (读取 label.txt) ===
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = []

        # 读取 label.txt
        label_file = os.path.join(root_dir, "label.txt")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"找不到标签文件: {label_file}")

        # 第一遍：收集所有类别名称并排序，建立索引
        raw_labels = []
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    raw_labels.append(parts[1].lower())

        self.classes = sorted(list(set(raw_labels)))  # ['bird', 'car', 'cat'...]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print(f"检测到 {len(self.classes)} 个类别: {self.classes}")

        # 保存类别到文件，供 tools.py 使用
        if not os.path.exists("models"): os.makedirs("models")
        with open(CLASS_SAVE_PATH, 'w', encoding='utf-8') as f:
            for cls in self.classes:
                f.write(f"{cls}\n")

        # 第二遍：加载数据路径和标签索引
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0] + ".png"  # 假设图片是 png，根据实际情况调整
                    # 如果文件名里已经带了后缀，就不要加 .png
                    if os.path.exists(os.path.join(root_dir, parts[0])):
                        filename = parts[0]

                    img_path = os.path.join(root_dir, filename)
                    label_name = parts[1].lower()

                    if os.path.exists(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# === 3. 训练流程 ===
def train():
    print(f"🚀 开始训练 Dataset B，使用设备: {DEVICE}")

    # 数据增强
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 标准输入
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(DATASET_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 加载预训练 ResNet18
    model = models.resnet18(weights='DEFAULT')

    # === 关键步骤：修改全连接层 ===
    # 原始 ResNet18 输出 1000 类，我们要改为 dataset 的类别数 (通常是10)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(dataset.classes))

    model = model.to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {running_loss / len(dataloader):.4f} | Acc: {acc:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ 模型已保存至: {MODEL_SAVE_PATH}")
    print(f"✅ 类别文件已保存至: {CLASS_SAVE_PATH}")


if __name__ == "__main__":
    train()