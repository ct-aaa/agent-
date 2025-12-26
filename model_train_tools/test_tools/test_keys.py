import torch
import os

# === 核心修改：自动获取脚本所在目录，确保 100% 能找到文件 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_PATH = os.path.join(current_dir, "pre_train", "model_best_TUBerlin.pth")


def check_pth_keys():
    print(f"📂 正在尝试读取: {PRETRAINED_PATH}")

    if not os.path.exists(PRETRAINED_PATH):
        print(f"❌ 依然找不到文件! 请检查路径是否包含中文或特殊字符。")
        return

    print(f"✅ 文件已找到，正在分析...")
    try:
        # 尝试加载
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')  # 暂时去掉 weights_only 以防报错

        state_dict = checkpoint

        # 1. 检查是否嵌套
        if isinstance(checkpoint, dict):
            print(f"📦 这是一个字典，包含的 Keys: {list(checkpoint.keys())}")
            if 'state_dict' in checkpoint:
                print("   👉 发现 'state_dict' 字段，正在提取参数...")
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                print("   👉 发现 'model' 字段，正在提取参数...")
                state_dict = checkpoint['model']
            # 如果没有 state_dict，那可能本身就是参数字典，继续往下走

        # 2. 打印前 10 个 Key
        print("\n🔑 --- 权重文件里的 Key (前10个) ---")
        if isinstance(state_dict, dict):
            keys = list(state_dict.keys())
            for k in keys[:10]:
                print(f"   {k}")
            print(f"\n📊 总参数量: {len(keys)}")
        else:
            print("❌ 加载出来的对象不是字典，无法读取参数名。")

    except Exception as e:
        print(f"❌ 读取出错: {e}")


if __name__ == "__main__":
    check_pth_keys()