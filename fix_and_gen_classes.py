import torch
import os

# --- TU-Berlin 标准 250 类别列表 (按字母顺序) ---
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


def fix_model_and_generate_txt():
    # 配置路径
    input_path = "models/model_c.pth"  # 你下载的原始文件
    output_model_path = "models/model_c_fixed.pth"  # 修复后的模型文件
    output_txt_path = "models/model_c_classes.txt"  # 自动生成的类别文件

    print(f"🚀 开始处理...")

    # ---------------------------------------------------------
    # 任务 1: 生成 classes.txt
    # ---------------------------------------------------------
    try:
        print(f"1️⃣ 正在生成类别文件: {output_txt_path}")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for cls_name in TU_BERLIN_CLASSES:
                f.write(cls_name + "\n")
        print(f"   ✅ 成功写入 {len(TU_BERLIN_CLASSES)} 个类别。")
    except Exception as e:
        print(f"   ❌ 写入失败: {e}")

    # ---------------------------------------------------------
    # 任务 2: 修复模型权重 (去除 module. 前缀)
    # ---------------------------------------------------------
    print(f"2️⃣ 正在修复权重文件: {input_path}")
    if not os.path.exists(input_path):
        print(f"   ❌ 错误：找不到原模型文件 {input_path}，请确认文件名。")
        return

    try:
        # 加载到 CPU 避免显存问题
        state_dict = torch.load(input_path, map_location="cpu", weights_only=False)

        new_state_dict = {}
        fixed_count = 0

        for k, v in state_dict.items():
            name = k
            if k.startswith("module."):
                name = k[7:]  # 移除 "module."
                fixed_count += 1
            new_state_dict[name] = v

        # 检查输出层维度
        if "fc.weight" in new_state_dict:
            out_features = new_state_dict["fc.weight"].shape[0]
            print(f"   ℹ️ 模型输出层维度检测: {out_features}")
            if out_features != len(TU_BERLIN_CLASSES):
                print(f"   ⚠️ 警告：模型输出维度({out_features})与列表长度({len(TU_BERLIN_CLASSES)})不一致！")

        torch.save(new_state_dict, output_model_path)
        print(f"   ✅ 权重修复完成，修正了 {fixed_count} 个参数名。")
        print(f"   💾 已保存至: {output_model_path}")

    except Exception as e:
        print(f"   ❌ 修复模型时出错: {e}")

    print("\n✨ 全部完成！请确保 evaluate_all.py 加载的是 'model_c_fixed.pth'")


if __name__ == "__main__":
    # 确保 models 文件夹存在
    os.makedirs("models", exist_ok=True)
    fix_model_and_generate_txt()