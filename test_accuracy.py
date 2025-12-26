import os
import tools
from tqdm import tqdm
import unicodedata


# 辅助函数：计算字符串的“显示宽度”（中文算2，英文算1）
def get_display_width(s):
    width = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ('F', 'W'):
            width += 2
        else:
            width += 1
    return width


# 辅助函数：生成固定显示宽度的字符串（用于对齐）
def pad_string(s, width):
    s = str(s)
    current_width = get_display_width(s)
    padding = width - current_width
    if padding > 0:
        return s + " " * padding
    return s  # 如果超出就不填充了，保持原样或按需截断


def calculate_accuracy(dataset_name):
    print(f"\n{'=' * 80}\n📊 测试: {dataset_name}\n{'=' * 80}")

    # 1. 获取图片
    image_list = tools.list_images(dataset_name)
    if not image_list:
        print("❌ 没找到图片")
        return

    # 2. 准备真值
    gt_map = {}
    if dataset_name != 'dataset_C':
        lbl_path = os.path.join("datasets", dataset_name, "label.txt")
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 2: gt_map[p[0]] = p[1].lower()

    correct = 0
    total = len(image_list)

    # 调整了列宽，pred 预留 40 字符
    header = f"| {pad_string('文件名', 15)} | {pad_string('真实值', 10)} | {pad_string('预测值', 40)} | {pad_string('判定', 6)} |"
    print(header)
    print("-" * get_display_width(header))  # 分割线长度自动匹配

    for img_path in tqdm(image_list, leave=False):  # leave=False 跑完后清除进度条
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]

        # 获取真值
        if dataset_name == 'dataset_C':
            ground_truth = os.path.basename(os.path.dirname(img_path)).lower()
        else:
            ground_truth = gt_map.get(stem, "???")

        # 预测
        pred = tools.classify_image(img_path)
        pred = str(pred).lower().strip()  # 去除首尾空格

        # --- 针对截图问题的特殊处理 ---
        # 如果预测结果里包含 "的识别结果是"，看起来比较冗余，你可以选择只显示关键部分
        # 如果不需要清洗，可以直接用 pred
        # clean_pred = pred.split(":")[-1].strip() if ":" in pred else pred

        # 判定
        if ground_truth == "???":
            mark = "❓"
        elif ground_truth == pred or ground_truth in pred or pred in ground_truth:
            correct += 1
            mark = "✅"
        else:
            mark = "❌"

        # 格式化输出
        # 1. 限制文件名长度防止太长，但保留足够长度
        d_fname = (fname[:12] + '..') if len(fname) > 14 else fname
        d_gt = (ground_truth[:8] + '..') if len(ground_truth) > 10 else ground_truth

        # 2. 预测值：根据你的截图，这个可能很长，我们放宽限制到 38 字符，超长才省略
        d_pred = (pred[:38] + '..') if len(pred) > 40 else pred

        # 3. 使用 pad_string 进行对齐
        row_str = f"| {pad_string(d_fname, 15)} | {pad_string(d_gt, 10)} | {pad_string(d_pred, 40)} | {pad_string(mark, 6)} |"

        # 4. 关键修改：使用 tqdm.write 代替 print
        tqdm.write(row_str)

    print("-" * get_display_width(header))
    if total > 0:
        print(f"🏁 准确率: {correct / total * 100:.2f}% ({correct}/{total})")
    else:
        print("🏁 图片数量为 0")


if __name__ == "__main__":
    # 可以单独注释掉某行来测试
    calculate_accuracy("dataset_A")
    calculate_accuracy("dataset_B")
    calculate_accuracy("dataset_C")