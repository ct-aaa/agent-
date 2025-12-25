import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


# === 1. 定义网络结构 (Dataset A - MNIST) ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# === 2. 模型缓存与加载 ===
_MODELS = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(dataset_name):
    if dataset_name in _MODELS: return _MODELS[dataset_name]

    try:
        if dataset_name == 'dataset_A':
            print("📥 加载 MNIST 模型 (dataset_A)...")
            model = SimpleCNN().to(device)
            # 修复 Warning: 添加 weights_only=True
            model.load_state_dict(torch.load("models/model_a.pth", map_location=device, weights_only=True))
            model.eval()
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
            # Dataset C 反色处理 (如果需要)
            # if dataset_name == 'dataset_C': img = ImageOps.invert(img)

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
        # 必须先运行 init_model_labels.py 生成这些 txt 文件
        label_file = {
            'dataset_A': 'models/model_a_classes.txt',
            'dataset_B': 'models/model_b_classes.txt',
            'dataset_C': 'models/model_c_classes.txt'
        }.get(dataset_name)

        predicted_label = str(class_id)  # 默认只返回 ID

        if label_file and os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines()]
                if class_id < len(classes):
                    predicted_label = classes[class_id]

        # 严格遵守接口定义：只返回类别字符串 (例如 "shark", "7")
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