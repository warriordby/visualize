def visualize_segmentation(prediction, save_path, colormap=None):
    """
    将语义分割预测图不同类别分配不同颜色进行可视化。

    参数:
    - prediction (np.array): 语义分割预测图，二维数组，值为类别ID。
    - save_path (str): 保存可视化图像的路径。
    - colormap (dict): 类别ID与颜色的映射字典，格式为 {类别ID: (R, G, B)}。
                       若为None，则自动生成随机颜色。

    返回:
    - None
    """
    colormap = {
        0: (255, 0, 0),       # 红色
        1: (0, 255, 0),       # 绿色
        2: (0, 0, 255),       # 蓝色
        3: (255, 255, 0),     # 黄色
        4: (0, 255, 255),     # 青色
        5: (255, 0, 255),     # 品红色
        6: (128, 0, 0),       # 深红色
        7: (0, 128, 0),       # 深绿色
        8: (0, 0, 128),       # 深蓝色
        9: (128, 128, 0),     # 橄榄色
        10: (0, 128, 128),    # 青绿色
        11: (128, 0, 128),    # 紫色
        12: (192, 192, 192),  # 银色
        13: (128, 128, 128),  # 灰色
        14: (255, 165, 0)     # 橙色
    }

    # 根据类别分配颜色
    color_image = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for cls, color in colormap.items():
        color_image[prediction == cls] = color

    # 保存彩色可视化图像
    cv2.imwrite(save_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
    print(f"可视化图像已保存至 {save_path}")