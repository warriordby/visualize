def sod_visualization(pred, gt, output_path, threshold=0.005):
    """
    可视化预测标签与真实标签不一致的区域。
    预测与真实标签不一致的区域将在真实标签图像上用红色填充，
    差异是基于预测与真实标签之间的差值阈值进行高亮显示。
    
    参数：
    - pred: 形状为 (B, H, W) 的张量，预测标签（经过sigmoid和阈值处理后）
    - gt: 形状为 (B, H, W) 的张量，真实标签
    - output_path: 输出图像保存路径
    - threshold: 用于高亮显示不匹配的差异阈值
    """
    # 将预测结果上采样到与真实标签相同的形状
    pred = F.interpolate(pred.float().unsqueeze(0).unsqueeze(0), size=gt.shape, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    pred = pred.sigmoid()  # 对预测结果应用sigmoid进行概率缩放
    
    # 将预测值归一化到0和1之间
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    mae = torch.sum(torch.abs(pred - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        
    # 将预测和真实标签转换为numpy数组以便处理
    pred = pred.cpu().numpy()  # 转换为numpy数组以便处理
    gt = gt.cpu().numpy()  # 转换为numpy数组以便处理

    # 创建一个与真实标签图像相同的副本用于可视化（3个通道）
    visual_image = np.repeat(gt[:, :, np.newaxis], 3, axis=-1) * 255  # 转换为3通道以便上色
    
    # 计算预测与真实标签之间的绝对差异
    diff = np.abs(pred - gt)

    # 将每次结果写入日志文件
    with open('/root/autodl-tmp/visual/sod_visual.txt','a') as f:
        f.write(output_path+' : '+str(mae)+'\n')
        
    # 创建掩码，标记差异超过阈值的区域,防止浮点数精度误差
    mismatch_mask = (diff > threshold)  
    
    # 对于不匹配的区域，将其颜色设置为红色
    visual_image[mismatch_mask] = [255, 0, 0]  # 不匹配的像素用红色表示
    
    # 确保visual_image是uint8格式，值范围在0到255之间
    visual_image = visual_image.astype(np.uint8)
    
    # 绘制并保存结果
    # plt.figure(figsize=(8, 8))
    plt.imshow(visual_image)
