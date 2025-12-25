import os
import torch
from torchvision import transforms, models
from PIL import Image

# --- 全局配置 ---
# 检测是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "models"  # 模型存放的文件夹名称

# 全局缓存：避免每次识别图片都重新读取硬盘加载模型，那会非常慢
model_cache = {}
class_cache = {}

# 预处理：必须与训练时(train_custom.py)保持完全一致
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_model_for_dataset(dataset_name):
    """
    核心函数：根据数据集名称加载对应的模型和类别表
    """
    # 如果缓存里已经有这个模型了，直接返回，不用重新加载
    if dataset_name in model_cache:
        return model_cache[dataset_name], class_cache[dataset_name]

    # 定义文件名映射关系 (请确保你的文件名和这里一致)
    config = {
        "dataset_A": {"model": "model_a.pth", "classes": "model_a_classes.txt"},
        "dataset_B": {"model": "model_b.pth", "classes": "model_b_classes.txt"},
        "dataset_C": {"model": "model_c.pth", "classes": "model_c_classes.txt"}
    }

    if dataset_name not in config:
        print(f"Error: 未知的数据集名称 {dataset_name}")
        return None, None

    # 构建完整路径
    model_path = os.path.join(MODELS_DIR, config[dataset_name]["model"])
    class_path = os.path.join(MODELS_DIR, config[dataset_name]["classes"])

    # 检查文件是否存在
    if not os.path.exists(model_path) or not os.path.exists(class_path):
        print(f"Error: 找不到模型或类别文件 -> {model_path} 或 {class_path}")
        return None, None

    print(f"正在加载模型: {dataset_name} ...")

    # 1. 读取类别文件
    with open(class_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    # 2. 初始化模型结构 (ResNet18)
    try:
        model = models.resnet18(weights=None)
        # 修改全连接层，使其输出节点数等于我们的类别数
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

        # 加载训练好的权重
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

        # 转移到 GPU/CPU 并开启评估模式
        model.to(device)
        model.eval()

        # 存入缓存
        model_cache[dataset_name] = model
        class_cache[dataset_name] = class_names

        return model, class_names

    except Exception as e:
        print(f"Error: 加载模型失败 {e}")
        return None, None


def list_images(dataset_name: str):
    """
    列出指定数据集的所有图片路径。
    """
    base_path = os.path.join("datasets", dataset_name)
    if not os.path.exists(base_path):
        return f"Error: 文件夹 {dataset_name} 不存在。"

    # 这里我们要确保返回的是完整的相对路径，例如 'datasets/dataset_B/1.png'
    # 这样后续 classify_image 才能直接读取
    images = []
    for f in os.listdir(base_path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join("datasets", dataset_name, f)
            images.append(full_path)

    # 按文件名数字大小排序 (可选，方便人类阅读)
    # images.sort()

    return images


def classify_image(image_path: str):
    """
    对单张图片进行分类。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(image_path):
        return f"Error: 图片文件不存在 {image_path}"

    # 2. 从路径中推断它是哪个数据集 (A, B 或 C)
    # Windows 路径可能是 datasets\dataset_B\1.png，也可能是 /
    normalized_path = image_path.replace("\\", "/")

    if "dataset_A" in normalized_path:
        ds_name = "dataset_A"
    elif "dataset_B" in normalized_path:
        ds_name = "dataset_B"
    elif "dataset_C" in normalized_path:
        ds_name = "dataset_C"
    else:
        return "Error: 无法从路径判断数据集来源 (文件名需包含 dataset_A/B/C)"

    # 3. 获取模型和类别表
    model, class_names = load_model_for_dataset(ds_name)
    if model is None:
        return f"Error: 模型 {ds_name} 加载失败"

    # 4. 图片预处理与推理
    try:
        # 打开图片并转为 RGB (防止有灰度图或 RGBA 图报错)
        img = Image.open(image_path).convert('RGB')
        # 增加 Batch 维度: [3, 224, 224] -> [1, 3, 224, 224]
        input_tensor = transform(img).unsqueeze(0).to(device)

        # 关掉梯度计算，节省内存
        with torch.no_grad():
            outputs = model(input_tensor)
            # 获取概率最大的那个类别的索引
            _, predicted_idx = torch.max(outputs, 1)

        # 将索引转换为文字标签
        predicted_label = class_names[predicted_idx.item()]
        return predicted_label

    except Exception as e:
        return f"Error: 识别过程出错 {e}"


def get_image_data(image_path: str):
    """
    获取指定路径的图片，用于最终显示。
    """
    return f"Image Displayed: {image_path}"
