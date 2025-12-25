import os
import tools
from tqdm import tqdm


def load_ground_truth(dataset_name):
    """
    读取 datasets/dataset_name/label.txt 文件
    返回一个字典: {'文件名(无后缀)': '真实标签'}
    例如: {'0': '7', '11': '6'}
    """
    label_path = os.path.join("datasets", dataset_name, "label.txt")
    ground_truth_map = {}

    if not os.path.exists(label_path):
        print(f"⚠️ 警告: 未找到标签文件 {label_path}，无法计算准确率。")
        return None

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line: continue

                # 分割每行数据 (默认按空格或tab分割)
                # 格式: "文件名索引  标签" -> ["11", "6"]
                parts = line.split()
                if len(parts) >= 2:
                    # key = 文件名 (例如 '11'), value = 标签 (例如 '6')
                    filename_stem = parts[0]
                    label = parts[1]
                    ground_truth_map[filename_stem] = label.lower()

        print(f"📄 已加载 {len(ground_truth_map)} 条真值数据 (来自 label.txt)")
        return ground_truth_map

    except Exception as e:
        print(f"❌ 读取标签文件出错: {e}")
        return None


def calculate_accuracy(dataset_name):
    print(f"\n{'=' * 70}")
    print(f"📊 测试数据集: {dataset_name}")
    print(f"{'=' * 70}")

    # 1. 获取所有图片路径
    image_list = tools.list_images(dataset_name)
    if not image_list:
        print(f"❌ 未找到图片或路径错误: datasets/{dataset_name}")
        return

    # 2. 加载真值表 (label.txt)
    gt_map = load_ground_truth(dataset_name)

    total = len(image_list)
    correct = 0

    print(f"共扫描到 {total} 张图片，正在推理...\n")

    # === 表格表头 ===
    # {filename:<15} 表示左对齐，占15个字符宽
    header = f"| {'文件名':<10} | {'真实值':<10} | {'预测结果':<25} | {'判定':<6} |"
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)

    for img_path in tqdm(image_list, leave=False):
        # --- 获取文件信息 ---
        file_name_full = os.path.basename(img_path)  # 例如 "11.png"
        file_stem = os.path.splitext(file_name_full)[0]  # 例如 "11"

        # --- 获取真值 ---
        # 如果没有 label.txt 或者找不到该图片的 key，标记为 "???"
        if gt_map and file_stem in gt_map:
            ground_truth = gt_map[file_stem]
        else:
            ground_truth = "???"

        # --- 调用工具预测 ---
        # tools.classify_image 返回的是字符串，例如 "7" 或 "shark"
        prediction = tools.classify_image(dataset_name, img_path)
        prediction_str = str(prediction).lower()

        # --- 判定逻辑 ---
        # 1. 如果没有真值，无法判定
        if ground_truth == "???":
            is_correct = False
            mark = "❓"  # 未知
        else:
            # 2. 字符串包含匹配 (适应 "shark" 匹配 "great white shark")
            #    或者完全相等 (适应 MNIST "7" == "7")
            if ground_truth == prediction_str or \
                    ground_truth in prediction_str or \
                    prediction_str in ground_truth:
                is_correct = True
                correct += 1
                mark = "✅"
            else:
                is_correct = False
                mark = "❌"

        # --- 表格行输出 (截断过长字符) ---
        f_disp = (file_name_full[:10])
        g_disp = (ground_truth[:10])
        p_disp = (prediction_str[:23] + '..') if len(prediction_str) > 23 else prediction_str

        print(f"| {f_disp:<10} | {g_disp:<10} | {p_disp:<25} | {mark:<6} |")

    # === 最终统计 ===
    print(divider)
    if gt_map:
        acc = (correct / total) * 100
        print(f"🏁 统计结果: 正确 {correct}/{total} | 准确率: {acc:.2f}%")
    else:
        print(f"⚠️ 无法计算准确率 (缺少标签文件)")


if __name__ == "__main__":
    # 你可以在这里切换想测的数据集
    #calculate_accuracy("dataset_A")
    calculate_accuracy("dataset_B")
    #calculate_accuracy("dataset_C")