import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

def CAM(features, img_path, save_path):
    # 假设 features 是模型输出的特征图，形状为 [1, C, H, W]
    features = F.interpolate(features, size=(400, 600), mode='bilinear', align_corners=False).squeeze(0)

    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    # 归一化 heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    # 使用颜色映射，将 heatmap 映射为 RGB
    color_map = plt.get_cmap('jet')
    heatmap = color_map(heatmap)[:, :, :3]  # 只取 RGB 通道
    heatmap = np.uint8(255 * heatmap)  # 将 heatmap 转换到 0-255 范围

    # 读取原始图像，并调整大小
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # 现在要确保只有热度较高的区域会应用热图，其他区域保留原始图像颜色
    alpha = 0.6  # 控制热图和原图混合的比例

    print(f"Heatmap Min Value: {np.min(heatmap)}")
    print(f"Heatmap Max Value: {np.max(heatmap)}")
    print(f"Heatmap Mean Value: {np.mean(heatmap)}")
    print(f"Heatmap Std Value: {np.std(heatmap)}")

    # 如果你想根据百分位数来设置阈值，比如设定为热图的90%分位数
    threshold = np.percentile(heatmap, 90)
    print(f"Threshold based on 90th percentile: {threshold}")

    # 使用该阈值来创建掩码
    mask = (heatmap > threshold)  # 这里的阈值 0.1 可以根据需求调整，控制热图的应用范围 
    
    masked_heatmap = np.zeros_like(heatmap)

    # 将高热度区域的热图值应用到新的热图
    masked_heatmap[mask] = heatmap[mask]

    # 叠加热图与原图像（仅高热度区域）
    superimposed_img = img_np * (1 - alpha) + masked_heatmap * alpha

    # 保证结果在合法范围内
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # 将最终的图像保存
    superimposed_img = Image.fromarray(superimposed_img)
    superimposed_img.save(save_path)

    print(f"Saved superimposed image at: {save_path}")


