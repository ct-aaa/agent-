import os
import torch
from torchvision import transforms, models
from PIL import Image

# --- 配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 关键修改 1: 定义不同的预处理 ---
# Dataset A (MNIST) 和 Dataset C (Sketch/ImageNet) 通常适用标准归一化
transform_standard = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset B (CIFAR-10) 必须使用与 train_standard.py 一致的归一化
transform_cifar = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # <--- 修正这里
])

# --- 关键修改 2: 同义词映射表 ---
# 格式: { "模型预测的词": "label.txt里的词" }
synonyms = {
    "automobile": "car",
    "plane": "airplane",  # 视情况而定，有时候 label是plane
    "airplane": "plane"  # 双向映射防止出错
}

# ... 保持前面的 import 和 transforms 配置不变 ...

# --- 内置 TU-Berlin 类别表 (作为自动修复的备份) ---
TU_BERLIN_CLASSES = [
    "airplane", "alarm clock", "ant", "ape", "apple", "arm", "armchair", "ashtray", "axe", "backpack",
    "banana", "barn", "baseball bat", "basket", "bathtub", "bear (animal)", "bed", "bee", "beer-mug", "bell",
    "bench", "bicycle", "binoculars", "blimp", "book", "bookshelf", "boomerang", "bottle opener", "bowl", "brain",
    "bread", "bridge", "bulldozer", "bus", "bush", "butterfly", "cabinet", "cactus", "cake", "calculator",
    "camel", "camera", "candle", "cannon", "canoe", "car (sedan)", "carrot", "castle", "cat", "cell phone",
    "chair", "chandelier", "church", "cigarette", "cloud", "comb", "computer monitor", "computer-mouse", "couch", "cow",
    "crab", "crane (machine)", "crocodile", "crown", "cup", "diamond", "dog", "dolphin", "donut", "door",
    "door handle", "dragon", "duck", "dumbbell", "ear", "elephant", "envelope", "eye", "eyeglasses", "face",
    "fan", "feather", "fence", "file cabinet", "fire hydrant", "fireplace", "firetruck", "fish", "flashlight",
    "floor lamp",
    "flower with stem", "flying bird", "flying saucer", "foot", "fork", "frog", "frying pan", "giraffe", "grapes",
    "grenade",
    "guitar", "hamburger", "hammer", "hand", "harp", "hat", "head", "headphones", "hedgehog", "helicopter",
    "helmet", "horse", "hot air balloon", "hot-dog", "hourglass", "house", "human-skeleton", "ice-cream-cone", "ipod",
    "kangaroo",
    "key", "keyboard", "knife", "ladder", "laptop", "leaf", "lightbulb", "lighter", "lion", "lobster",
    "loudspeaker", "mailbox", "megaphone", "mermaid", "microphone", "microscope", "monkey", "moon", "mosquito",
    "motorbike",
    "mouse (animal)", "mouth", "mug", "mushroom", "nose", "octopus", "owl", "palm tree", "panda", "paper clip",
    "parachute", "parking meter", "parrot", "pear", "pen", "penguin", "person sitting", "person walking", "piano",
    "pickup truck",
    "pig", "pigeon", "pineapple", "pipe (for smoking)", "pizza", "plane", "planet", "pocket watch", "postcard",
    "potato",
    "potted plant", "power outlet", "present", "pretzel", "pumpkin", "purse", "rabbit", "race car", "racket", "radio",
    "rainbow", "revolver", "rifle", "rollerblades", "rooster", "sailboat", "santa claus", "satellite", "satellite dish",
    "saxophone",
    "scissors", "scorpion", "screw", "screwdriver", "sea turtle", "seagull", "shark", "sheep", "ship", "shoe",
    "shovel", "skateboard", "skull", "skyscraper", "snail", "snake", "snowboard", "snowman", "socks", "space shuttle",
    "speed-boat", "spider", "sponge bob", "spoon", "squirrel", "standing bird", "stapler", "strawberry", "streetlamp",
    "submarine",
    "suitcase", "sun", "sunflower", "swan", "sword", "syringe", "table", "tablelamp", "teacup", "teapot",
    "teddy-bear", "telephone", "television", "tennis-racket", "tent", "tiger", "tire", "toilet", "tomato", "tooth",
    "toothbrush", "toothpaste", "tornado", "tractor", "traffic light", "train", "tree", "trombone", "trousers", "truck",
    "trumpet", "t-shirt", "tv", "umbrella", "van", "vase", "violin", "walkie talkie", "wheel", "wheelbarrow",
    "windmill", "wine-bottle", "wineglass", "wrist-watch", "zebra"
]


def ensure_classes_file(dataset_name, class_path):
    """如果类别文件不存在，尝试自动生成"""
    if os.path.exists(class_path):
        return True

    if dataset_name == "dataset_C":
        print(f"⚠️ 未找到 {class_path}，正在自动生成...")
        try:
            with open(class_path, "w", encoding="utf-8") as f:
                f.write("\n".join(TU_BERLIN_CLASSES))
            return True
        except Exception as e:
            print(f"无法生成类别文件: {e}")
            return False
    return False


def load_model(dataset_name):
    config = {
        "dataset_A": ("models/model_a.pth", "models/model_a_classes.txt"),
        "dataset_B": ("models/model_b.pth", "models/model_b_classes.txt"),
        "dataset_C": ("models/model_c.pth", "models/model_c_classes.txt")
    }

    if dataset_name not in config: return None, None
    model_path, class_path = config[dataset_name]

    # 1. 自动检查并生成类别文件
    if not ensure_classes_file(dataset_name, class_path):
        if not os.path.exists(class_path):
            print(f"❌ 缺少类别文件: {class_path}")
            return None, None

    if not os.path.exists(model_path):
        print(f"❌ 缺少模型文件: {model_path}")
        return None, None

    # 读取类别
    with open(class_path, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    # 2. 初始化模型架构
    if dataset_name == "dataset_C":
        model = models.resnet50(weights=None)
    else:
        model = models.resnet18(weights=None)

    # 修改全连接层
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

    # 3. 智能加载权重 (Smart Loading)
    try:
        # weights_only=False 消除警告
        state_dict = torch.load(model_path, map_location=device, weights_only=False)

        # 自动清洗 'module.' 前缀
        new_state_dict = {}
        cleaned_count = 0
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # 去掉前7个字符
                cleaned_count += 1
            else:
                new_state_dict[k] = v

        if cleaned_count > 0:
            print(f"ℹ️ 自动修正了 {cleaned_count} 个带有 'module.' 前缀的权重参数。")

        # 使用 strict=False 加载，但我们会打印不匹配的键来排查问题
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        if len(missing) > 0:
            # 过滤掉 fc 层的 mismatch，因为那可能是我们改过的，只要骨干网络加载了就行
            important_missing = [k for k in missing if "fc" not in k]
            if len(important_missing) > 0:
                print(
                    f"⚠️ 警告: 有 {len(important_missing)} 个关键层权重未加载 (可能导致准确率低): {important_missing[:5]}...")

    except Exception as e:
        print(f"❌ 模型加载严重错误: {e}")
        return None, None

    model.to(device)
    model.eval()
    return model, classes


def evaluate_dataset(dataset_name):
    print(f"\n正在评估数据集: {dataset_name} ...")

    dataset_dir = os.path.join("datasets", dataset_name)
    label_file = os.path.join(dataset_dir, "label.txt")

    if not os.path.exists(label_file):
        print("未找到 label.txt，跳过。")
        return

    # 加载模型
    model, classes = load_model(dataset_name)
    if not model: return

    # 选择对应的 transform
    if dataset_name == "dataset_B":
        current_transform = transform_cifar
    else:
        current_transform = transform_standard

    correct = 0
    total = 0

    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2: continue

        img_id, true_label = parts[0], parts[1]
        img_path = os.path.join(dataset_dir, f"{img_id}.png")

        if not os.path.exists(img_path): continue

        # 预测
        try:
            img = Image.open(img_path).convert('RGB')
            img_t = current_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_t)
                pred_idx = torch.max(out, 1)[1].item()
                pred_label = classes[pred_idx]

            # --- 核心逻辑: 比较时考虑同义词 ---
            is_correct = False
            # 1. 直接相等
            if pred_label.lower() == true_label.lower():
                is_correct = True
            # 2. 查同义词表 (例如 pred是automobile, table里有automobile->car, 且true_label是car)
            elif pred_label.lower() in synonyms and synonyms[pred_label.lower()] == true_label.lower():
                is_correct = True
            # 3. 反向查表 (例如 pred是car, table里有car->automobile)
            elif true_label.lower() in synonyms and synonyms[true_label.lower()] == pred_label.lower():
                is_correct = True

            total += 1
            if is_correct:
                correct += 1
            else:
                print(f"  [错] 图片 {img_id}: 预测={pred_label}, 真实={true_label}")

        except Exception as e:
            print(f"  处理错误 {img_id}: {e}")

    acc = 100 * correct / total if total > 0 else 0
    print(f"📊 {dataset_name} 准确率: {correct}/{total} ({acc:.2f}%)")


if __name__ == "__main__":
    evaluate_dataset("dataset_A")
    evaluate_dataset("dataset_B")
    evaluate_dataset("dataset_C")