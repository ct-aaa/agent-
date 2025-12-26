import pickle
import numpy as np
import os
import matplotlib

# 设置后端为 Agg (非交互式)，这比默认后端更快且不需要显示器支持
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 忽略 numpy 版本警告
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_strokes_from_data(vector_data):
    """
    辅助函数：解析数据并拆分笔画（乱线修复逻辑）
    """
    strokes_list = []

    # 情况 1: 数据已经是列表
    if isinstance(vector_data, list):
        strokes_list = vector_data

    # 情况 2: Numpy 数组 (N, 3) 包含抬笔信息
    elif isinstance(vector_data, np.ndarray):
        if vector_data.ndim == 2 and vector_data.shape[1] >= 3:
            current_stroke = []
            for point in vector_data:
                x, y, pen_state = point[0], point[1], point[2]
                current_stroke.append([x, y])
                if pen_state != 0:  # 抬笔
                    strokes_list.append(np.array(current_stroke))
                    current_stroke = []
            if len(current_stroke) > 0:
                strokes_list.append(np.array(current_stroke))
        else:
            # 纯 (N, 2) 无法拆分
            strokes_list = [vector_data]

    return strokes_list


def process_one_image(args):
    """
    工作进程函数：处理单张图片
    Args 包含: (key, vector_data, target_root_dir)
    """
    key, vector_data, target_root_dir = args

    try:
        # 解析路径
        try:
            category, file_id = key.split('/')
        except ValueError:
            category = "misc"
            file_id = key

        # 确保目录存在 (多进程中需要注意竞态条件，但 exist_ok=True 通常没问题)
        class_dir = os.path.join(target_root_dir, category)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, f"{file_id}.png")

        # 如果文件已存在，跳过（可选，为了速度）
        # if os.path.exists(save_path): return

        # === 绘图 ===
        # 创建新的 figure 实例而不是使用 plt 接口，这样在多线程/进程中更安全
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes([0, 0, 1, 1])  # 占满画布
        ax.set_axis_off()

        strokes_list = get_strokes_from_data(vector_data)

        for stroke in strokes_list:
            stroke = np.array(stroke)
            if stroke.size > 0:
                if stroke.ndim == 1: stroke = stroke.reshape(-1, 2)
                if len(stroke) > 1:
                    # 翻转 Y 轴
                    ax.plot(stroke[:, 0], -stroke[:, 1], 'k-', linewidth=2)

        # 保存
        fig.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.05)

        # 显式清理
        plt.close(fig)
        return True  # 成功

    except Exception as e:
        print(f"Error processing {key}: {e}")
        return False


def load_and_convert_parallel(source_file, target_root_dir):
    """
    主函数：准备数据并启动并行处理
    """
    if not os.path.exists(source_file):
        print(f"找不到文件: {source_file}")
        return

    print("正在加载庞大的 Pickle 文件到内存 (这一步是单核的，请耐心等待)...")
    with open(source_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    total_files = len(data)
    print(f"加载完成，准备并行处理 {total_files} 张图像...")

    # 准备任务列表
    # 我们把所有需要的数据打包成元组列表
    tasks = []
    for key, val in data.items():
        tasks.append((key, val, target_root_dir))

    # 确定 CPU 核心数
    # 如果为了防止卡死电脑，可以设置为 cpu_count() - 2
    num_workers = max(1, multiprocessing.cpu_count())
    print(f">>> 启动 {num_workers} 个进程全速运行 CPU <<<")

    # 启动进程池
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 使用 list(tqdm(...)) 来触发迭代并显示进度条
        results = list(tqdm(executor.map(process_one_image, tasks), total=total_files, unit="img"))

    print(f"\n全部处理完成！文件保存在: {target_root_dir}")


if __name__ == "__main__":
    # Windows 下必须把逻辑放在 if __name__ == "__main__": 之下
    SOURCE_PATH = os.path.join('..', 'datasets', 'TU_Berlin')
    TARGET_DIR = os.path.join('..', 'datasets', 'dataset_TU')

    load_and_convert_parallel(SOURCE_PATH, TARGET_DIR)