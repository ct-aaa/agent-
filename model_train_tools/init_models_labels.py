import os
import urllib.request

# 定义保存目录
MODEL_DIR = "../models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"📁 已创建目录: {MODEL_DIR}")


def generate_mnist_classes():
    """生成 MNIST (Model A) 的类别文件: 0-9"""
    filename = os.path.join(MODEL_DIR, "model_a_classes.txt")
    print(f"正在生成 {filename} ...")

    with open(filename, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(f"{i}\n")
    print("✅ MNIST 类别文件生成完毕。")


def generate_imagenet_classes():
    """
    下载并生成 ImageNet (Model B 和 C) 的类别文件
    ResNet 和 MobileNet 默认权重都是在 ImageNet-1k 上训练的
    """
    # PyTorch 官方使用的 ImageNet 类别映射表 (纯英文)
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

    # 我们为 Model B (ResNet) 和 Model C (MobileNet) 生成相同的文件
    targets = ["model_b_classes.txt", "model_c_classes.txt"]

    print(f"正在从 {url} 下载标准 ImageNet 标签...")

    try:
        # 下载数据
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')

        # 写入文件
        for target in targets:
            filepath = os.path.join(MODEL_DIR, target)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ 已生成: {filepath}")

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("请检查网络连接，或者手动搜索 'imagenet_classes.txt' 填入文件中。")


if __name__ == "__main__":
    print("🚀 开始初始化模型标签文件...")
    generate_mnist_classes()
    print("-" * 30)
    generate_imagenet_classes()
    print("-" * 30)
    print("🎉 所有标签文件准备就绪！")