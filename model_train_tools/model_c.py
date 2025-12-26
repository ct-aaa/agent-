import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps, ImageFilter
import os
import glob

# ================= ⚙️ 配置区域 =================
# 1. 你的“考试题” (只有一张图的数据集)
DATA_C_DIR = "../datasets/dataset_C"

# 2. 你的“教科书” (TU-Berlin 完整数据集)
DATA_TU_DIR = "../datasets/dataset_TU"

# 3. 权重路径
PRETRAINED_PATH = "pre_train/model_best_TUBerlin.pth"

# 4. 保存位置
MODEL_SAVE_DIR = "../models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_model_trained_on_TU_tested_on_C.pth")

BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================

# === 1. 视觉增强预处理 (保持不变，因为效果很好) ===
class SketchEnhancement(object):
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = ImageOps.invert(img)  # 反转颜色
        fn = lambda x: 255 if x > 50 else 0
        img = img.convert('L').point(fn, mode='1').convert('RGB')  # 二值化
        # img = img.filter(ImageFilter.MaxFilter(3)) # 膨胀 (TU数据本身线条较好，可根据情况开关)
        return img


# === 2. 自定义数据集类：只加载 TU-Berlin 中指定的类别 ===
class FilteredTUDataset(Dataset):
    def __init__(self, tu_root_dir, target_classes, transform=None):
        """
        tu_root_dir: TU-Berlin 根目录
        target_classes: 我们关心的那 20 个类别的名字列表 ['bed', 'bee', ...]
        """
        self.transform = transform
        self.samples = []  # 存储 (图片路径, 标签ID)
        self.classes = target_classes

        # 建立 类别名 -> ID 的映射 (确保和 dataset_C 一致)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(target_classes)}

        print(f"🔍 正在从 TU-Berlin 筛选数据...")
        count = 0
        for class_name in target_classes:
            class_dir = os.path.join(tu_root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"   ⚠️ 警告: TU-Berlin 中找不到类别 {class_name}，跳过！")
                continue

            # 找所有图片
            images = glob.glob(os.path.join(class_dir, "*.png")) + \
                     glob.glob(os.path.join(class_dir, "*.jpg"))

            label_idx = self.class_to_idx[class_name]
            for img_path in images:
                self.samples.append((img_path, label_idx))
                count += 1

        print(f"✅ 筛选完成！共加载 {count} 张训练图片 (涵盖 {len(target_classes)} 类)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# === 权重加载函数 ===
def load_github_weights_fixed(model, path):
    if not os.path.exists(path): return model
    print(f"📥 加载基础权重: {path}")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['state_dict'] if (
                isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('Network.features.', '').replace('module.', '')
        if 'classifier' not in k and 'fc' not in name:
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    return model


def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # --- 1. 获取目标类别列表 (以 dataset_C 为准) ---
    if not os.path.exists(DATA_C_DIR):
        print("❌ 找不到 dataset_C")
        return

    # 自动读取 dataset_C 下的文件夹名作为目标类别
    target_classes = sorted([d for d in os.listdir(DATA_C_DIR) if os.path.isdir(os.path.join(DATA_C_DIR, d))])
    print(f"🎯 目标类别 ({len(target_classes)}): {target_classes}")

    # --- 2. 准备数据增强 ---
    # 训练集 (TU数据): 加强扰动，让模型见过世面
    train_transform = transforms.Compose([
        SketchEnhancement(),
        transforms.Resize(256),
        transforms.RandomCrop(224),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机翻转
        transforms.RandomRotation(15),  # 随机旋转 (很重要，增加泛化性)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 验证集 (C数据): 保持稳定，只做必要的缩放
    val_transform = transforms.Compose([
        SketchEnhancement(),
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 3. 构建数据集 ---
    # 训练集：来自 TU-Berlin (筛选版)
    train_dataset = FilteredTUDataset(DATA_TU_DIR, target_classes, transform=train_transform)

    # 验证集：来自 Dataset_C (全量验证)
    val_dataset = datasets.ImageFolder(DATA_C_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 4. 模型初始化 ---
    print("🛠️ 初始化模型...")
    model = models.resnet50(weights=None)
    model = load_github_weights_fixed(model, PRETRAINED_PATH)

    # 替换最后一层为 20 类
    model.fc = nn.Linear(model.fc.in_features, len(target_classes))
    model = model.to(DEVICE)

    # --- 5. 训练 ---
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print(f"👊 开始跨域训练 (Train: TU-Berlin -> Test: Dataset_C)...")

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证 (考试)
        model.eval()
        correct = 0
        total = 0
        debug_log = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                # 记录一下预测错误的样本，方便看
                if i == 0:
                    # 只看第一个 batch 里的错题
                    wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]
                    for idx in wrong_idx:
                        if len(debug_log) < 3:  # 只记前3个
                            true_cls = target_classes[labels[idx].item()]
                            pred_cls = target_classes[preds[idx].item()]
                            debug_log.append(f"❌ 错把 [{true_cls}] 认成 [{pred_cls}]")

        val_acc = 100 * correct / total
        avg_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch + 1:02d} | Train Loss: {avg_loss:.4f} | 🎯 Dataset_C Acc: {val_acc:.2f}%")
        if debug_log:
            print(f"   错题本: {debug_log}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   🏆 新高分！模型已保存。")

    print(f"🎉 结束！最佳成绩: {best_acc:.2f}%")


if __name__ == "__main__":
    main()