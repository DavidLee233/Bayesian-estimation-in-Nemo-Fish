'''
该函数主要负责获取对应Nemo鱼的mask（通过RGB贝叶斯决策）
'''
from numpy import ma
import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.stats import multivariate_normal, norm


def get_train_data():
	# 加载训练样本
	array_sample = scio.loadmat('array_sample.mat')
	array_sample = array_sample['array_sample']

	# 加载测试样本mask
	Mask = scio.loadmat('Mask.mat')
	mask = Mask['Mask'].astype('bool')

	# 获取背景部分的RGB色值
	nemo_309_color = cv2.imread('309.bmp', cv2.IMREAD_COLOR)
	nemo_309_color = cv2.cvtColor(nemo_309_color, cv2.COLOR_BGR2RGB)
	bg_data = nemo_309_color[~mask] / 255 # shape -> (N1, 3)
	fish_data = array_sample[:, 1:4] # shape -> (N2, 3)

	return bg_data, fish_data


# GMM判别函数
def rgb_discriminant_func(pixels, mu1, sigma1, mu2, sigma2, w_1, w_2, w1=1, w2=1):
	'''
    pixels:[N, 3], 像素点的rgb值,N=WxH
    return: labels [N, 1]，像素点的标签<-1 or 1>
    '''
    # 计算等效后验概率
	p_1 = multivariate_normal.pdf(pixels, mu1, sigma1)* w_1 * w1
	p_2 = multivariate_normal.pdf(pixels, mu2, sigma2)* w_2 * w2

	labels = np.ones_like(p_1)
	index_2 = p_2 > p_1
	labels[index_2] = -1
	return labels

# 腐蚀膨胀
def Morphological_operation(same_out, k=1):
	# 定义卷积核
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
	kernel = np.ones([5, 5])
	img = same_out
	for _ in range(k):

		# img = cv2.erode(img,kernel)
		# #显示膨胀后的图像
		# # cv2.imshow("Dilated Image",img)
		img = cv2.dilate(img,kernel)
		#显示膨胀后的图像
		# cv2.imshow("Dilated Image",img)
		# img = cv2.dilate(img,kernel)
	
	return img


def get_mask(img_name):

	if img_name == '317.bmp':
		img_w = 170
	else:
		img_w = 174

	origin_img = cv2.imread(img_name, cv2.IMREAD_COLOR)
	img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

	# 截断分类（方便后面腐蚀膨胀）
	img = img[:img_w, 40:220]

	bg_data, fish_data = get_train_data()
	# 估计均值和方差
	bg_mu = np.mean(bg_data, axis=0)
	bg_sigma = np.cov(bg_data, rowvar=False, bias=True)

	fish_mu = np.mean(fish_data, axis=0)
	fish_sigma = np.cov(fish_data, rowvar=False, bias=True)

	# 估计类别先验概率
	w_1 = bg_data.shape[0] / (bg_data.shape[0] + fish_data.shape[0])
	w_2 = 1 - w_1

	origin_size = img.shape[:2]
	out = rgb_discriminant_func(np.reshape(img  / 255, (-1, 3)), bg_mu, bg_sigma,
								fish_mu, fish_sigma, w_1, w_2)

	same_out = np.reshape(out, origin_size) # (240, 320)
	# set bool mask
	same_out[same_out == 1] = 0
	same_out[same_out == -1] = 1
	same_out = np.array(same_out, dtype=np.uint8)
	same_out = Morphological_operation(same_out)
	final_mask = np.zeros([240, 320])
	final_mask[:img_w, 40:220] = same_out
	
	return final_mask


