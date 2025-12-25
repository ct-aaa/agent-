import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- 全局配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 模型文件路径配置
config = {
    "dataset_A": {"model": "model_a.pth", "classes": "model_a_classes.txt"},
    "dataset_B": {"model": "model_b.pth", "classes": "model_b_classes.txt"},
    # 确保这里的文件名和你 fine_tune_pro.py 保存的一致
    "dataset_C": {"model": "model_c_pro.pth", "classes": "model_c_classes.txt"}
}

# 2. 模型架构配置
MODEL_ARCH_CONFIG = {
    "dataset_A": "resnet18",
    "dataset_B": "resnet18",
    "dataset_C": "mobilenet_v3_small"
}

_MODEL_CACHE = {}

# 3. 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model_for_dataset(dataset_name):
    """加载模型并缓存"""
    if dataset_name in _MODEL_CACHE:
        return _MODEL_CACHE[dataset_name]

    if dataset_name not in config:
        return None, []

    info = config[dataset_name]
    model_path = os.path.join("models", info["model"])
    txt_path = os.path.join("models", info["classes"])

    if not os.path.exists(model_path) or not os.path.exists(txt_path):
        # 仅在第一次找不到文件时打印，方便调试
        print(f"❌ 错误: 找不到文件 {model_path} 或 {txt_path}")
        return None, []

    try:
        # 1. 读取类别表
        with open(txt_path, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]

        # 2. 加载权重字典
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict'] if (
                    isinstance(checkpoint, dict) and 'state_dict' in checkpoint) else checkpoint

        # 3. 清洗权重键名 (去掉 module. 等前缀)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            name = name.replace('module.', '').replace('Network.features.', '').replace('Network.classifier.', 'fc.')
            new_state_dict[name] = v

        # 4. 确定类别数量
        # 优先从权重形状判断，如果没有fc.weight则使用类别表长度
        if 'fc.weight' in new_state_dict:
            model_num_classes = new_state_dict['fc.weight'].shape[0]
        elif 'classifier.3.1.weight' in new_state_dict:  # 针对 MobileNet 的情况
            model_num_classes = new_state_dict['classifier.3.1.weight'].shape[0]
        else:
            model_num_classes = len(classes)

        # 5. 初始化模型架构 (核心修复部分)
        arch = MODEL_ARCH_CONFIG.get(dataset_name, "resnet18")

        if arch == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, model_num_classes)

        elif arch == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=None)
            # MobileNet 必须修改 classifier，而不是 fc
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = nn.Sequential(
                nn.Dropout(p=0.3),  # 保持和训练时一致的结构，虽然推理时 dropout 不生效
                nn.Linear(num_ftrs, model_num_classes)
            )

        else:  # 默认 ResNet18
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, model_num_classes)

        # ⚠️ 注意：这里不再有 model.fc = ... 的通用代码了！

        # 6. 加载参数
        # strict=False 可以容忍一些不匹配，比如 dropout 层名字差异，防止报错
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()

        _MODEL_CACHE[dataset_name] = (model, classes)
        return model, classes

    except Exception as e:
        # 这里打印具体的错误信息，方便你看到真正的报错
        print(f"❌ 加载模型 {dataset_name} 崩溃: {e}")
        import traceback
        traceback.print_exc()
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
    if not model: return "Error: 模型加载失败 (查看上方报错)"

    try:
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
            prob = torch.nn.functional.softmax(out[0], dim=0)
            score, idx = torch.max(prob, 0)

            if idx.item() >= len(classes):
                return f"Error: 索引越界"

            class_name = classes[idx.item()]
            confidence = score.item() * 100

            return f"{class_name} ({confidence:.1f}%)"

    except Exception as e:
        return f"Error: {e}"