import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from line_kernel import create_line_kernel

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def create_strel_config():
    """创建结构元素配置 - 使用列表存储参数"""
    return [
        # 每个子列表代表一个腐蚀序列: [名称, [(长度1, 角度1), (长度2, 角度2)], 显示标签]
        ["co1", [(5, -45), (7, -45)], "串联1"],
        ["co2", [(5, 45), (7, 45)], "串联2"],
        ["co3", [(3, 90), (5, 90)], "串联3"],
        ["co4", [(3, 0), (5, 0)], "串联4"],
    ]


def apply_erosion_sequence(image, strel_config):
    """应用腐蚀序列 - 返回腐蚀结果列表和标签列表"""
    erosion_results = []  # 存储最终腐蚀结果
    erosion_labels = []  # 存储对应的显示标签

    for config in strel_config:
        _, kernel_params, label = config
        current = image.copy()

        # 应用序列中的每个结构元素
        for length, degree in kernel_params:
            kernel = create_line_kernel(length, degree)
            current = cv2.erode(current, kernel)

        erosion_results.append(current)
        erosion_labels.append(label)

    return erosion_results, erosion_labels


def calculate_weights(noisy_image, erosion_results):
    """计算各腐蚀结果的权重 - 返回权重列表和总权重"""
    weights = []
    noisy = noisy_image.astype(np.float32)

    for result in erosion_results:
        # 计算绝对差异和
        diff = np.sum(np.abs(result.astype(np.float32) - noisy))
        weights.append(diff)

    total_weight = sum(weights)
    return weights, total_weight


def combine_results(erosion_results, weights, total_weight):
    """合并去噪结果 - 加权平均"""
    if total_weight == 0:
        return np.zeros_like(erosion_results[0])

    combined = np.zeros_like(erosion_results[0], dtype=np.float32)

    for i, result in enumerate(erosion_results):
        weight = weights[i] / total_weight
        combined += weight * result.astype(np.float32)

    # 归一化到[0,1]范围
    return (combined - np.min(combined)) / (np.max(combined) - np.min(combined))


def psnr(original, processed):
    """计算峰值信噪比"""
    # 转换为0-255范围
    original_norm = cv2.normalize(original, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    processed_norm = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # 计算均方误差
    mse = np.mean((original_norm.astype(float) - processed_norm.astype(float)) ** 2)
    if mse == 0:
        return float("inf")

    # 计算PSNR
    max_val = 255.0
    return 10 * np.log10(max_val**2 / mse)


# ====================== 图像处理流程 ======================
def load_and_preprocess_image(filepath):
    """加载并预处理图像"""
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {filepath}")

    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def add_poisson_noise(image):
    """添加泊松噪声"""
    noisy = np.random.poisson(image / 255.0 * 100) / 100.0 * 255.0
    return noisy.astype(np.uint8)


# ====================== 可视化函数 ======================
def plot_images(original, noisy, erosion_results, erosion_labels, combined):
    """绘制图像结果"""
    # 原始图像和噪声图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("原图像")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(noisy, cmap="gray")
    plt.title("噪声图像")
    plt.axis("off")

    # 各腐蚀结果
    plt.figure(figsize=(12, 12))
    for i, (result, label) in enumerate(zip(erosion_results, erosion_labels)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(result, cmap="gray")
        plt.title(label)
        plt.axis("off")

    # 噪声图像和最终结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(noisy, cmap="gray")
    plt.title("噪声图像")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(combined, cmap="gray")
    plt.title("并联去噪图像")
    plt.axis("off")


def plot_psnr(original, erosion_results, erosion_labels, combined):
    """绘制PSNR曲线"""
    psnr_values = []
    labels = []

    # 各腐蚀结果的PSNR
    for result, label in zip(erosion_results, erosion_labels):
        psnr_val = psnr(original, result)
        psnr_values.append(psnr_val)
        labels.append(label)

    # 最终结果的PSNR
    psnr_values.append(psnr(original, combined))
    labels.append("并联")

    # 绘制曲线
    plt.figure()
    plt.plot(range(1, len(psnr_values) + 1), psnr_values, "r+-")

    # 自动调整Y轴范围
    min_psnr = min(psnr_values) - 1
    max_psnr = max(psnr_values) + 1
    plt.axis([0, len(psnr_values) + 1, min_psnr, max_psnr])

    # 设置X轴标签
    x_labels = [""] + labels + [""]
    plt.xticks(range(0, len(psnr_values) + 2), x_labels)

    plt.grid(True)
    plt.title("PSNR曲线比较")
    plt.ylabel("PSNR (dB)")
    plt.tight_layout()


# ====================== 主流程 ======================
def main():
    # 加载并预处理图像
    filename = "im.jpg"
    original_image = load_and_preprocess_image(filename)

    # 添加噪声
    noisy_image = add_poisson_noise(original_image)

    # 创建结构元素配置
    strel_config = create_strel_config()

    # 应用腐蚀序列
    erosion_results, erosion_labels = apply_erosion_sequence(noisy_image, strel_config)

    # 计算权重并合并结果
    weights, total_weight = calculate_weights(noisy_image, erosion_results)
    combined_result = combine_results(erosion_results, weights, total_weight)

    # 可视化结果
    plot_images(
        original_image, noisy_image, erosion_results, erosion_labels, combined_result
    )

    # 计算并绘制PSNR
    plot_psnr(original_image, erosion_results, erosion_labels, combined_result)

    plt.show()


if __name__ == "__main__":
    main()
