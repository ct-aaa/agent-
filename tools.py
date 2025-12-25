import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


# === 1. 更新后的网络结构 (需与训练脚本一致) ===
class BetterCNN(nn.Module):
    def __init__(self):
        super(BetterCNN, self).__init__()

        # 第一层卷积块: Conv -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 新增层

        # 第二层卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # 新增层

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)  # 新增层
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
        # 注意：推理时 model.eval() 会自动关闭 dropout，这里保留结构即可
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# === 2. 模型缓存与加载 ===
_MODELS = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(dataset_name):
    if dataset_name in _MODELS: return _MODELS[dataset_name]

    try:
        if dataset_name == 'dataset_A':
            print("📥 加载 MNIST 优化版模型 (dataset_A)...")
            # --- 修改点：实例化 BetterCNN ---
            model = BetterCNN().to(device)
            # 加载参数
            model.load_state_dict(torch.load("models/model_a.pth", map_location=device, weights_only=True))
            model.eval()  # 关键！这会关闭 Dropout 和 BatchNorm 的训练模式
            _MODELS[dataset_name] = model

        elif dataset_name == 'dataset_B':
            print("📥 加载 ResNet18 (dataset_B)...")
            model = models.resnet18(weights='DEFAULT').to(device)
            model.eval()
            _MODELS[dataset_name] = model

        elif dataset_name == 'dataset_C':
            print("📥 加载 MobileNetV3 (dataset_C)...")
            model = models.mobilenet_v3_small(weights='DEFAULT').to(device)
            model.eval()
            _MODELS[dataset_name] = model

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    return _MODELS.get(dataset_name)


# === 3. 核心分类函数 ===
def classify_image(dataset_name, image_path):
    model = get_model(dataset_name)
    if not model: return "Error: Model not loaded"

    try:
        img = Image.open(image_path)

        # 预处理
        if dataset_name == 'dataset_A':
            tf = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            img = img.convert('RGB')
            tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        img_t = tf(img).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            out = model(img_t)
            prob = torch.nn.functional.softmax(out[0], dim=0)
            score, idx = torch.max(prob, 0)
            class_id = idx.item()

        # 映射类别 ID -> 名称
        label_file = {
            'dataset_A': 'models/model_a_classes.txt',
            'dataset_B': 'models/model_b_classes.txt',
            'dataset_C': 'models/model_c_classes.txt'
        }.get(dataset_name)

        predicted_label = str(class_id)

        if label_file and os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines()]
                if class_id < len(classes):
                    predicted_label = classes[class_id]

        return predicted_label

    except Exception as e:
        return f"Error: {str(e)}"


# === 4. 工具函数 ===
def list_images(dataset_name):
    path = os.path.join("datasets", dataset_name)
    if not os.path.exists(path): return []
    images = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, f).replace('\\', '/'))
    return images


def get_image_data(image_path):
    return image_path