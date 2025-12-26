import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps
import os


# === 1. MNIST 网络结构 (Dataset A) ===
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


# === 2. 模型加载器 (单例模式) ===
_MODELS = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(dataset_key):
    if dataset_key in _MODELS: return _MODELS[dataset_key]

    try:
        # --- 加载 MNIST ---
        if dataset_key == 'dataset_A':
            model = BetterCNN().to(device)
            if os.path.exists("models/model_a.pth"):
                model.load_state_dict(torch.load("models/model_a.pth", map_location=device, weights_only=True))
            model.eval()
            _MODELS[dataset_key] = model

        # --- 加载 CIFAR-10 ---
        elif dataset_key == 'dataset_B':
            try:
                hub_dir = r"C:\Users\admin\.cache\torch\hub\chenyaofo_pytorch-cifar-models_master"
                if os.path.exists(hub_dir):
                    model = torch.hub.load(hub_dir, "cifar10_resnet20", pretrained=True, source='local')
                else:
                    model = torch.hub.load("chenyaofo/pytorch-cifar-pre_train", "cifar10_resnet20", pretrained=True)
            except:
                # 兜底：直接下载
                model = torch.hub.load("chenyaofo/pytorch-cifar-pre_train", "cifar10_resnet20", pretrained=True)

            model = model.to(device)
            model.eval()
            _MODELS[dataset_key] = model

        # --- 加载 Dataset C (素描) ---
        elif dataset_key == 'dataset_C':
            # 优先加载最强模型
            pth_file = "models/best_model_trained_on_TU_tested_on_C.pth"
            if not os.path.exists(pth_file):
                pth_file = "models/best_model_dataset_c.pth"

            # 读取类别数
            num_classes = 20
            if os.path.exists("models/classes_c.txt"):
                with open("models/classes_c.txt", 'r', encoding='utf-8') as f:
                    num_classes = len([l for l in f.readlines() if l.strip()])

            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            if os.path.exists(pth_file):
                ckpt = torch.load(pth_file, map_location=device, weights_only=False)
                # 处理 state_dict 嵌套
                sd = ckpt['state_dict'] if (isinstance(ckpt, dict) and 'state_dict' in ckpt) else ckpt
                model.load_state_dict(sd, strict=False)

            model.to(device)
            model.eval()
            _MODELS[dataset_key] = model

    except Exception as e:
        print(f"❌ 模型 {dataset_key} 加载失败: {e}")
        return None
    return _MODELS.get(dataset_key)


# === 3. 核心工具: 图片分类 (含数据绑定) ===
def classify_image(image_path):
    """
    只需提供图片路径，自动判断数据集并调用对应模型。
    返回格式: "[文件名] 的识别结果是: 类别" (防止 LLM 看错行)
    """
    # 路径标准化
    path_str = image_path.replace('\\', '/')
    filename = os.path.basename(path_str)  # 提取文件名 "1.png"

    # 1. 自动路由
    if 'dataset_A' in path_str:
        key = 'dataset_A'
    elif 'dataset_B' in path_str:
        key = 'dataset_B'
    elif 'dataset_C' in path_str:
        key = 'dataset_C'
    else:
        return f"Error: 无法识别数据集来源 {filename}"

    model = get_model(key)
    if not model: return f"Error: 模型初始化失败 {filename}"

    try:
        img = Image.open(image_path)

        # 2. 针对性预处理 (特别是 Dataset C 的视觉增强)
        if key == 'dataset_A':
            tf = transforms.Compose([
                transforms.Grayscale(1), transforms.Resize((28, 28)),
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif key == 'dataset_B':
            img = img.convert('RGB')
            tf = transforms.Compose([
                transforms.Resize((32, 32)), transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
        else:  # Dataset C (关键：必须反转+二值化)
            img = img.convert('RGB')
            img = ImageOps.invert(img)  # 反转
            fn = lambda x: 255 if x > 50 else 0
            img = img.convert('L').point(fn, mode='1').convert('RGB')  # 二值化

            tf = transforms.Compose([
                transforms.Resize(240), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        img_t = tf(img).unsqueeze(0).to(device)

        # 3. 推理
        with torch.no_grad():
            out = model(img_t)
            idx = torch.max(out, 1)[1].item()

        # 4. 标签解码
        if key == 'dataset_B':
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            lbl = classes[idx]
            if lbl == 'automobile': lbl = 'car'
            if lbl == 'airplane': lbl = 'plane'
        else:
            txt = "models/model_a_classes.txt" if key == 'dataset_A' else "models/classes_c.txt"
            if os.path.exists(txt):
                with open(txt, 'r', encoding='utf-8') as f:
                    cls = [x.strip() for x in f.readlines() if x.strip()]
                lbl = cls[idx] if idx < len(cls) else str(idx)
            else:
                lbl = str(idx)

        # ⚠️ 关键修改：把文件名和结果绑定在一起返回
        return f"[{filename}] 的识别结果是: {lbl}"

    except Exception as e:
        return f"Error processing {filename}: {e}"


# === 4. 工具: 列出图片 (含排序) ===
def list_images(dataset_name):
    path = os.path.join("datasets", dataset_name)
    if not os.path.exists(path): return []
    res = []
    for r, _, fs in os.walk(path):
        for f in fs:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                res.append(os.path.join(r, f).replace('\\', '/'))
    # ⚠️ 关键修改：强制排序，保证 Agent 每次看到的顺序一致
    res.sort()
    return res


# === 5. 工具: 计算器 (升级版：支持比较大小) ===
def calculate(expression):
    """
    参数: expression (str), 例如 "1+2+3", "12 > 8" 或 "10 % 3"
    功能: 精确计算数学表达式的结果，支持取模运算
    """
    try:
        # 允许数字、运算符号 (+-*/)、括号、比较符号 (><=) 以及 取模 (%)
        allowed = set("0123456789+-*/(). ><=%")

        # 检查是否包含非法字符
        if not all(c in allowed for c in expression):
            return "Error: 包含非法字符，拒绝计算"

        # 使用 eval 进行计算 (eval 本身支持 % 运算)
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: 计算出错 {e}"