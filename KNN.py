import scipy.io as scio
import numpy as np
import cv2
from get_mask import get_mask
## 距离函数
# 欧氏距离
def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2, axis=1))

# 曼哈顿距离
def manhattan(x, y):
    return np.sum(np.abs(x - y), axis=1)

# KNN算法
def KNN(konwed_data, X, k=11, dis_method='e'):
	'''
	konwed_data: 已知标签的样本特征数据；gray -> (N, 2); rgb -> (N, 4)；第最后一列表示标签
	X: 待分类的样本数据; gray -> (n, 1); rgb -> (n, 3)
	k: 自定义的邻域范围：选取与待分类点最近的k个已知样本
	dis_method: 距离度量函数，默认欧氏距离，指定为'm'变为曼哈顿距离。可选参数['e', 'm']

	return:
		labels: 返回每个样本点分类后的label,标签-1 or 1; shape -> (n,) 
	'''
	# 计算先验概率
	w_1 = np.sum(konwed_data[:, -1] == 1) / konwed_data.shape[0]
	w_2 = 1 - w_1
	# print(w_1, w_2)
	# 设置风险权重
	w1 = 1
	w2 = 1
	
	labels = np.zeros(X.shape[0])
	# 计算距离
	for i, x in enumerate(X):

		if dis_method == 'e':
		# 采用欧氏距离
			dis = euclidean(x, konwed_data[:, :-1])
		else:
			dis = manhattan(x, konwed_data[:, :-1])
		idx = np.argsort(dis)
		k_nerbor = konwed_data[idx[:k]] # 取距离前k个的样本

		count_1 = np.sum(k_nerbor[:, -1] == 1)
		count_2 = k - count_1

		p_1 = count_1 / k * w_1 * w1
		p_2 = count_2 / k * w_2 * w2

		if p_1 > p_2:
			labels[i] = -1
		else:
			labels[i] = 1
	
	return labels


# 加载训练样本
array_sample = scio.loadmat('array_sample.mat')
array_sample = array_sample['array_sample']


img_name = '317.bmp'
if '309' in img_name:
	# 加载测试样本mask
	Mask = scio.loadmat('Mask.mat')
	mask = Mask['Mask']
else:
	mask = get_mask(img_name)

# 获取测试样本图像
nemo_rgb = cv2.imread(img_name, cv2.IMREAD_COLOR)
nemo_rgb = cv2.cvtColor(nemo_rgb, cv2.COLOR_BGR2RGB)
nemo_gray = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

# 获取带标签的灰度样本数据
konwed_data_gray = array_sample[:, [0, 4]]
origin_size_gray = nemo_gray.shape
X_gray = np.reshape(nemo_gray  / 255, (-1))

# 进行灰度knn分类
gray_labels = KNN(konwed_data_gray, X_gray)
nemo_gray_out = np.reshape(gray_labels, origin_size_gray) * mask

# 为分类后的Nemo鱼上色
nemo_gray_out[nemo_gray_out == 1] = 100
nemo_gray_out[nemo_gray_out == -1] = 255
nemo_gray_out = np.array(nemo_gray_out, dtype=np.uint8)

# 获取带标签的彩色样本数据
konwed_data_rgb = array_sample[:, 1:]
origin_size_rgb = nemo_rgb.shape
X_rgb = np.reshape(nemo_rgb  / 255, (-1, 3))

# 进行彩色knn分类
rgb_labels = KNN(konwed_data_rgb, X_rgb)

# 为分类后的Nemo鱼上色
nemo_rgb_out_2d = np.reshape(rgb_labels, origin_size_rgb[:2]) * mask # (240, 320)
nemo_rgb_out = np.expand_dims(nemo_rgb_out_2d,2).repeat(3,axis=2) # (240, 320, 3)

nemo_rgb_out[nemo_rgb_out_2d == -1] = np.array([255, 0, 0], dtype=np.uint8)
nemo_rgb_out[nemo_rgb_out_2d == 1]= np.array([255, 255, 255], dtype=np.uint8)
nemo_rgb_out[nemo_rgb_out_2d == 0] = np.array([0, 0, 0], dtype=np.uint8)

# 可视化
mask[mask == 1] = 255
cv2.imshow('Nemo_mask', mask)
cv2.imshow('Nemo_gray_origin', nemo_gray)
cv2.imshow('Nemo_rgb_origin', cv2.cvtColor(nemo_rgb, cv2.COLOR_BGR2RGB))
cv2.imshow('Nemo_gray_out', nemo_gray_out)
cv2.imshow('Nemo_rgb_out', nemo_rgb_out)
cv2.waitKey(0)
cv2.destroyAllWindows()	


		
