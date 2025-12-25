import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# --- 第一步：定义自定义数据集类 ---
class CustomTextDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.images = []  # 存储图片文件名
        self.labels = []  # 存储标签字符串

        # 读取 label.txt
        label_file = os.path.join(dataset_dir, "label.txt")
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"找不到标签文件: {label_file}")

        print(f"正在读取数据集: {dataset_dir} ...")

        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 解析每一行，例如 "1 bird"
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_id = parts[0]  # 例如 "1"
                label_name = parts[1]  # 例如 "bird"

                # 拼接图片文件名，假设图片格式是 .png
                img_filename = f"{img_id}.png"

                # 检查图片是否存在
                if os.path.exists(os.path.join(dataset_dir, img_filename)):
                    self.images.append(img_filename)
                    self.labels.append(label_name)

        # 创建 类别 -> 数字索引 的映射 (例如: {'bird': 0, 'car': 1})
        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        print(f"发现类别: {self.class_to_idx}")
        print(f"总图片数: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label_str = self.labels[idx]

        # 读取图片
        img_path = os.path.join(self.dataset_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 获取数字标签
        label_idx = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label_idx


# --- 第二步：训练函数 ---
def train_one_dataset(data_dir, save_path):
    # 1. 设置数据预处理
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 加载数据集
    dataset = CustomTextDataset(data_dir, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 3. 建立模型 (使用预训练的 ResNet18)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # 修改最后一层以匹配当前数据集的类别数
    num_classes = len(dataset.classes)
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 5. 简单的训练循环 (训练 5 个 epoch)
    print("开始训练...")
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.4f}")

    # 6. 保存模型和类别映射
    # 保存权重
    torch.save(model.state_dict(), save_path)

    # 同时保存一个对应的 txt 文件，记录这个模型能识别哪些类别顺序
    # (例如 model_b_classes.txt 内容为: bird car cat ...)
    classes_path = save_path.replace('.pth', '_classes.txt')
    with open(classes_path, 'w') as f:
        f.write('\n'.join(dataset.classes))

    print(f"模型已保存至: {save_path}")
    print(f"类别表已保存至: {classes_path}\n" + "-" * 30)


# --- 主程序入口 ---
if __name__ == "__main__":
    # 确保 models 文件夹存在
    os.makedirs("models", exist_ok=True)

    # 训练 Dataset B (你需要针对 A 和 C 也做同样的操作)
    # 假设你的文件夹结构是 ./datasets/dataset_B
    train_one_dataset("./datasets/dataset_B", "models/model_b.pth")
    train_one_dataset("./datasets/dataset_A", "models/model_a.pth")
    train_one_dataset("./datasets/dataset_C", "models/model_c.pth")
