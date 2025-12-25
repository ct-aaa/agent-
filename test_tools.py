import tools
import os


def test_tools():
    print("========== 开始测试 Tools 功能 ==========")

    # 1. 测试 list_images
    print("\n[测试 1] list_images ('dataset_B')")
    images = tools.list_images('dataset_A')

    if isinstance(images, list) and len(images) > 0:
        print(f"✅ 成功获取列表，共找到 {len(images)} 张图片。")
        print(f"   第一张图片路径: {images[0]}")
    else:
        print(f"❌ 获取列表失败: {images}")

    # 2. 测试 classify_image (抽取第一张图测试)
    if images and len(images) > 0:
        test_img_path = images[1]  # 通常是 datasets\dataset_B\1.png
        print(f"\n[测试 2] classify_image ('{test_img_path}')")

        # 调用分类函数
        result = tools.classify_image(test_img_path)
        print(f"👉 分类结果: {result}")

        # 验证是否是硬编码 (Mock)
        # 如果你还没有修改 tools.py，这里可能会一直返回同一个词
        if result == "bird" or result == "7":
            print("⚠️ 警告: 如果你换了不同的图片测试结果还是一样，说明 tools.py 可能还在使用示例代码，未加载真实模型。")

    # 3. 测试不存在的文件 (鲁棒性测试)
    print("\n[测试 3] 测试不存在的文件")
    fake_result = tools.classify_image("datasets/non_existent.png")
    print(f"👉 错误处理返回: {fake_result}")


if __name__ == "__main__":
    test_tools()
