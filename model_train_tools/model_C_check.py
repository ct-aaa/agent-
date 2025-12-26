# 简单的统计脚本，看看是不是数据不平衡导致了倾向于预测 "dog"
from collections import Counter

label_file = "../datasets/dataset_E/label.txt"  # 假设路径
try:
    with open(label_file, 'r') as f:
        labels = [line.strip().split(' ')[1] for line in f.readlines()]  # 假设格式是 filename,label

    counts = Counter(labels)
    print("类别分布:", counts)
    print("最常见类别:", counts.most_common(3))
except Exception as e:
    print(f"无法读取: {e}")