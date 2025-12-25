import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps  # 引入 ImageOps 用于反色
import os

# --- 全局配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 模型文件配置
config = {
    "dataset_A": {"model": "model_a.pth", "classes": "model_a_classes.txt"},
    "dataset_B": {"model": "model_b.pth", "classes": "model_b_classes.txt"},
    "dataset_C": {"model": "model_c.pth", "classes": "model_c_classes.txt"}
}

# 2. 架构配置
MODEL_ARCH_CONFIG = {
    "dataset_A": "resnet18",
    "dataset_B": "resnet18",
    "dataset_C": "resnet50"
}

_MODEL_CACHE = {}

# 3. 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model_for_dataset(dataset_name):
    if dataset_name in _MODEL_CACHE:
        return _MODEL_CACHE[dataset_name]

    if dataset_name not in config:
        return None, []

    info = config[dataset_name]
    model_path = os.path.join("models", info["model"])
    txt_path = os.path.join("models", info["classes"])

    if not os.path.exists(model_path) or not os.path.exists(txt_path):
        print(f"❌ 文件缺失: {model_path} 或 {txt_path}")
        return None, []

    try:
        # 读取类别
        with open(txt_path, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]

        # 加载权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict'] if (
                    isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint

        # 清洗 Key
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith('module.'): name = name[7:]
            if name.startswith('Network.features.'): name = name.replace('Network.features.', '')
            if name.startswith('Network.classifier.'): name = name.replace('Network.classifier.', 'fc.')
            new_state_dict[name] = v

        # 智能判断类别数
        if 'fc.weight' in new_state_dict:
            model_num_classes = new_state_dict['fc.weight'].shape[0]
        else:
            model_num_classes = len(classes)

        # 初始化模型
        arch = MODEL_ARCH_CONFIG.get(dataset_name, "resnet18")
        if arch == "resnet50":
            model = models.resnet50(weights=None)
        else:
            model = models.resnet18(weights=None)

        model.fc = nn.Linear(model.fc.in_features, model_num_classes)
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()

        _MODEL_CACHE[dataset_name] = (model, classes)
        return model, classes

    except Exception as e:
        print(f"❌ 加载模型失败 {dataset_name}: {e}")
        return None, []


def list_images(dataset_name):
    path = os.path.join("datasets", dataset_name)
    if not os.path.exists(path): return f"Error: {path} not found"
    images = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(root, f).replace('\\', '/'))
    return images


def classify_image(image_path):
    if "dataset_A" in image_path:
        ds = "dataset_A"
    elif "dataset_B" in image_path:
        ds = "dataset_B"
    elif "dataset_C" in image_path:
        ds = "dataset_C"
    else:
        return "Error: 路径中未包含 dataset_A/B/C"

    model, classes = load_model_for_dataset(ds)
    if not model: return "Error: 模型加载失败"

    try:
        img = Image.open(image_path).convert('RGB')

        # # === 🚑 关键修复：针对 Dataset_C 的自动反色 ===
        # if ds == "dataset_C":
        #     # 简单采样判断亮度：如果左上角是白色的(255)，说明是白底黑线，需要反色
        #     # 或者直接计算平均亮度
        #     from torchvision.transforms.functional import to_tensor
        #     if to_tensor(img).mean() > 0.5:
        #         # print("Detected white background, inverting...")
        #         img = ImageOps.invert(img)
        # # ==========================================

        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
            prob = torch.nn.functional.softmax(out[0], dim=0)
            score, idx = torch.max(prob, 0)

            if idx.item() >= len(classes):
                return f"Error: 索引越界"

            return f"{classes[idx.item()]} ({score.item() * 100:.1f}%)"
    except Exception as e:
        return f"Error: {e}"