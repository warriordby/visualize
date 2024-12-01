from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.cluster import KMeans



"""
该脚本实现了基于 t-SNE 降维后对图像数据进行 K-means 聚类，并可视化聚类结果和聚类中心。主要功能包括：
1. 图像数据的读取和预处理。
2. 使用 t-SNE 将高维图像数据降维至2D或3D空间。
3. 使用 K-means 聚类算法对降维后的数据进行聚类。
4. 可视化聚类结果和聚类中心，聚类中心的大小根据距离的最大值进行调整。

"""

# 该函数用于可视化聚类结果和聚类中心，3D可视化
def plot_centroid_visualization_3d(X, y, title=None, filename='centroid_visualization.png'):
    """
    可视化聚类中心及其大小，大小基于到聚类内点的最大距离。
    
    参数:
    X (ndarray): 降维后的数据，形状为 (n_samples, 3)，包含3D坐标
    y (ndarray): 标签数组，表示每个样本所属的类别
    title (str): 图表标题
    filename (str): 保存图像的文件名
    
    输出:
    生成并保存一个包含聚类中心的3D可视化图
    对每个类别可视化聚类分布范围，不同聚类中心使用不同的颜色区分
    """
    # K-means 聚类
    n_clusters = len(np.unique(y))  # 获取类别数
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    y_kmeans = kmeans.fit_predict(X)  # 聚类结果

    # 计算聚类中心
    centroids = kmeans.cluster_centers_

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 计算每个聚类的点到聚类中心的最大距离
    max_distances = []
    for i in range(len(centroids)):
        cluster_points = X[y_kmeans == i]
        if len(cluster_points) > 0:
            centroid = centroids[i]
            max_distance = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
            max_distances.append(max_distance)
        else:
            max_distances.append(0)

    # 创建颜色映射
    unique_labels = np.unique(y_kmeans)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    # 为每个类别绘制不同颜色的点
    for idx, label in enumerate(unique_labels):
        ax.scatter(X[y_kmeans == label, 0], X[y_kmeans == label, 1], X[y_kmeans == label, 2], 
                   color=colors[idx], s=1, alpha=0.8, label=f'Cluster {label}')

    # 绘制聚类中心，颜色与类别相同，大小基于最大距离
    for i in range(len(centroids)):
        ax.scatter(centroids[i, 0], centroids[i, 1], centroids[i, 2], 
                   s=max_distances[i] * 500,  # 根据最大距离调整大小
                   color=colors[i], alpha=0.5, label=f'Centroid {i}')

    # 创建自定义图例句柄
    legend_handles = []
    for i in range(len(centroids)):
        legend_handles.append(ax.scatter([], [], color=colors[i], s=50, alpha=0.8, label=f'Centroid {i}'))

    # 添加标题和图例
    if title is not None:
        plt.title(title)

    ax.legend(handles=legend_handles + 
              [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}', 
                           markerfacecolor=colors[idx], markersize=5) for idx, label in enumerate(unique_labels)], 
              loc='upper right', bbox_to_anchor=(1.35, 1))

    plt.savefig(filename)  # 保存图像
    plt.show()  # 显示图像
    plt.close()  # 关闭当前图形以释放内存


# 该函数用于可视化聚类结果和聚类中心，2D可视化
def plot_centroid_visualization_2d(X, y, title=None, filename='centroid_visualization.png'):
    """
    可视化聚类中心及其大小，大小基于到聚类内点的最大距离。
    
    参数:
    X (ndarray): 降维后的数据，形状为 (n_samples, 2)，包含2D坐标
    y (ndarray): 标签数组，表示每个样本所属的类别
    title (str): 图表标题
    filename (str): 保存图像的文件名
    
    输出:
    生成并保存一个包含聚类中心的3D可视化图
    对每个类别可视化聚类分布范围，不同聚类中心使用不同的颜色区分
    """
    # K-means 聚类
    n_clusters = len(np.unique(y))  # 获取类别数
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    y_kmeans = kmeans.fit_predict(X)  # 聚类结果

    # 计算聚类中心
    centroids = kmeans.cluster_centers_

    # 创建2D图形
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 计算每个聚类的点到聚类中心的最大距离
    max_distances = []
    for i in range(len(centroids)):
        cluster_points = X[y_kmeans == i]
        if len(cluster_points) > 0:
            centroid = centroids[i]
            max_distance = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
            max_distances.append(max_distance)
        else:
            max_distances.append(0)

    # 创建颜色映射
    unique_labels = np.unique(y_kmeans)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    # 为每个类别绘制不同颜色的点
    for idx, label in enumerate(unique_labels):
        ax.scatter(X[y_kmeans == label, 0], X[y_kmeans == label, 1], 
                   color=colors[idx], s=1, alpha=0.8, label=f'Cluster {label}')

    # 绘制聚类中心，颜色与类别相同，大小基于最大距离
    for i in range(len(centroids)):
        ax.scatter(centroids[i, 0], centroids[i, 1],
                   s=max_distances[i] * 500,  # 根据最大距离调整大小
                   color=colors[i], alpha=0.5, label=f'Centroid {i}')

    # 创建自定义图例句柄
    legend_handles = []
    for i in range(len(centroids)):
        legend_handles.append(ax.scatter([], [], color=colors[i], s=50, alpha=0.8, label=f'Centroid {i}'))

    # 添加标题和图例
    if title is not None:
        plt.title(title)

    ax.legend(handles=legend_handles + 
              [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label}', 
                           markerfacecolor=colors[idx], markersize=5) for idx, label in enumerate(unique_labels)], 
              loc='upper right', bbox_to_anchor=(1.35, 1))

    plt.savefig(filename)  # 保存图像
    plt.show()  # 显示图像
    plt.close()  # 关闭当前图形以释放内存


if __name__ == '__main__':


    # 图像文件名和路径
    img_list = ['1_1', '9_9','5_5','1_8','5_9']
    root_path = r'C:/Users/86156/Desktop/program/python/workspace/train/Image1'
    images = []

    # 读取图像并重塑为数组
    for img_name in img_list:
        path = os.path.join(root_path, img_name + '.png')
        img = Image.open(path)
        img = img.resize((32, 32), Image.LANCZOS)  # 调整为32x32大小
        images.append(np.array(img).reshape(-1, 3))  # 每个图像重塑为一维数组

    # 拼接所有图像数据
    X = np.concatenate(images, axis=0)


    # 设计标签 y
    y = np.array([i for i in range(len(img_list)) for _ in range(images[i].shape[0])])

    # 处理标签以包含额外类别 ###############################3
    extra_label = len(img_list)  # 新类别的标签
    pixel_labels = []

    for pixel in X:
        if np.sum(np.all(X == pixel, axis=1)) > 1:  # 检查是否有相同的像素
            pixel_labels.append(extra_label)
        else:
            pixel_labels.append(y[np.where((X == pixel).all(axis=1))[0][0]])
    # 转换为数组
    y = np.array(pixel_labels)


    # 打印形状以确认
    print(X.shape)

    print("Computing t-SNE embedding")
    tsne3d = TSNE(n_components=3, init='pca', random_state=0)
    tsne2d = TSNE(n_components=2, init='pca', random_state=0)


    X_tsne_3d = tsne3d.fit_transform(X)
    X_tsne_2d = tsne2d.fit_transform(X)


    # plot_centroid_visualization_3d(X_tsne_3d, y,"Cluster Centroids with Distances", "centroid_visualization3d.png")
    # 调用新函数以绘制聚类中心
    plot_centroid_visualization_2d(X_tsne_2d, y,"Cluster Centroids with Distances", "centroid_visualization2d.png")
