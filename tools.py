import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


# === 1. 这里的 BetterCNN 仅用于 Dataset A (MNIST) ===
class BetterCNN(nn.Module):
    def __init__(self):
        super(BetterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# === 2. 模型缓存 ===
_MODELS = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(dataset_name):
    if dataset_name in _MODELS: return _MODELS[dataset_name]

    try:
        if dataset_name == 'dataset_A':
            print("📥 加载 MNIST 模型 (dataset_A)...")
            model = BetterCNN().to(device)
            # 确保 weights_only=True 以避免警告
            if os.path.exists("models/model_a.pth"):
                model.load_state_dict(torch.load("models/model_a.pth", map_location=device, weights_only=True))
            else:
                print("⚠️ 警告: models/model_a.pth 不存在，请先训练 Model A")
            model.eval()
            _MODELS[dataset_name] = model

        elif dataset_name == 'dataset_B':

            print("📥 正在加载本地缓存的 CIFAR-10 模型 (离线模式)...")

            # 1. 设置你的本地缓存路径 (根据你的报错截图提取的路径)

            # 使用 r"" 防止反斜杠转义问题

            hub_dir = r"C:\Users\admin\.cache\torch\hub\chenyaofo_pytorch-cifar-models_master"

            if not os.path.exists(hub_dir):
                print(f"❌ 错误: 找不到本地缓存目录: {hub_dir}")
                print("请先用联网模式运行一次，或检查路径是否正确。")
                return None

            try:

                # 2. 核心修改: source='local'
                # 这告诉 PyTorch 不要去 GitHub 查更新，直接用硬盘里的文件
                model = torch.hub.load(hub_dir, "cifar10_resnet20", pretrained=True, source='local')
                model = model.to(device)
                model.eval()
                _MODELS[dataset_name] = model
                print("✅ 模型加载成功 (Local)")


            except Exception as e:
                print(f"❌ 本地加载失败: {e}")
                print("尝试检查 cache 文件夹里是否有 hubconf.py 文件")
                return None

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

        # === 针对不同数据集使用不同的预处理 ===
        if dataset_name == 'dataset_A':
            # MNIST: 28x28, 灰度
            tf = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif dataset_name == 'dataset_B':
            # CIFAR-10: 32x32, RGB, 标准化参数不同
            img = img.convert('RGB')
            tf = transforms.Compose([
                transforms.Resize((32, 32)),  # 关键：CIFAR 模型需要 32x32
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        else:
            # Dataset C (ImageNet): 224x224
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

        # === 结果映射 ===
        predicted_label = str(class_id)

        if dataset_name == 'dataset_B':
            # CIFAR-10 的类别是固定的，我们直接硬编码，不需要读 txt 文件
            # 这样更稳健
            cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            if class_id < len(cifar_classes):
                raw_label = cifar_classes[class_id]
                # 兼容性处理：把 standard label 转换成你 label.txt 里的叫法
                # 你的 label.txt 用的是 "car", "plane"
                if raw_label == 'automobile':
                    predicted_label = 'car'
                elif raw_label == 'airplane':
                    predicted_label = 'plane'
                else:
                    predicted_label = raw_label
        else:
            # 其他模型继续读取 txt
            label_file = {
                'dataset_A': 'models/model_a_classes.txt',
                'dataset_C': 'models/model_c_classes.txt'
            }.get(dataset_name)

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