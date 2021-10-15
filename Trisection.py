from scipy import io
from scipy.stats import norm
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
mask = io.loadmat('Mask.mat')['Mask']     # 数据为一个字典，根据key提取数据
sample = io.loadmat('array_sample.mat')['array_sample']
src_image = Image.open('311.bmp')
RGB_img = np.array(src_image) #将四张图像转换为多维数组类型以便计算
Gray_img = np.array(src_image.convert('L')) #将309.bmp彩色图像转换为灰度图像
RGB_mask = np.array([mask, mask, mask]).transpose(1, 2, 0)#将RGB图像的数据格式由（channels,imagesize,imagesize）转化为（imagesize,imag
RGB_Pro = RGB_img/255
Gray_Pro = Gray_img/255
gray1 = []
gray2 = []
gray3 = []
RGB1 = []
RGB2 = []
RGB3 = []
for i in range(len(sample)):
    if(sample[i][4]) == 1.:                          #数据第5位为标签
        RGB1.append(sample[i][1:4])  # 将RGB值提取
        gray1.append(sample[i][0])  # 将灰度值提取
    else:
        RGB2.append(sample[i][1:4])
        gray2.append(sample[i][0])
where_bg = np.where(mask==0)
RGB3.append(RGB_Pro[where_bg])
gray3.append(Gray_Pro[where_bg])
np.reshape(RGB3,(-1,3))
RGB1 = np.array(RGB1)
RGB2 = np.array(RGB2)
RGB3 = np.array(RGB3)
RGB3 = np.reshape(RGB3,(-1,3))
gray3 = np.reshape(gray3,(-1,1))
# gray3 = np.array(gray3)
# gray3 = np.array(gray3.tolist())
gray3 = gray3[:, 0]
gray3 = gray3.tolist()
# gray3 = np.float64(gray3)
# 计算图像两类在数据中的占比，即先验概率
P_pre1 = len(RGB1)/sum([len(RGB1),len(RGB2),len(RGB3)])
P_pre2 = len(RGB2)/sum([len(RGB1),len(RGB2),len(RGB3)])
P_pre3 = 1-P_pre1-P_pre2
# 一维时的贝叶斯决策
# ------------------------------------------------------------------------------------#
# 数据为一维时(灰度图像)，用最大似然分别估计灰度图像和RGB图像中两个类别条件概率pdf的参数——标准差与均值
gray1_m = np.mean(gray1)
gray2_m = np.mean(gray2)
gray3_m = np.mean(gray3)
gray1_s = np.std(gray1)
gray2_s = np.std(gray2)
gray3_s = np.std(gray3)
# 绘制灰度图像最大似然估计出的类条件pdf
x = np.arange(0, 1, 1/1000) #函数返回一个有终点和起点的固定步长的排列
gray1_pdf = norm.pdf(x, gray1_m, gray1_s) #x,x的均值,x的标准差来计算第一类的类条件概率密度
gray2_pdf = norm.pdf(x, gray2_m, gray2_s)
gray3_pdf = norm.pdf(x, gray3_m, gray3_s)
# 用最大后验概率贝叶斯决策对灰度图像进行分割
gray_out = np.zeros_like(Gray_img) #构建一个与Gray_img同维度的数组，并初始化所有变量为零
for i in range(len(Gray_Pro)):
    for j in range(len(Gray_Pro[0])):
        if Gray_Pro[i][j] == 0:
            continue
        elif max(P_pre1*norm.pdf(Gray_Pro[i][j], gray1_m, gray1_s),P_pre2*norm.pdf(Gray_Pro[i][j], gray2_m, gray2_s),P_pre3*norm.pdf(Gray_Pro[i][j], gray3_m, gray3_s)) == P_pre1*norm.pdf(Gray_Pro[i][j], gray1_m, gray1_s):   # 贝叶斯公式分子比较
            gray_out[i][j] = 100
        elif max(P_pre1*norm.pdf(Gray_Pro[i][j], gray1_m, gray1_s),P_pre2*norm.pdf(Gray_Pro[i][j], gray2_m, gray2_s),P_pre3*norm.pdf(Gray_Pro[i][j], gray3_m, gray3_s)) == P_pre2*norm.pdf(Gray_Pro[i][j], gray2_m, gray2_s):
            gray_out[i][j] = 255
        else:
            gray_out[i][j] = 0
plt.figure(0)
bx = plt.subplot(2, 2, 1)
bx.set_title('gray nemo')
bx.imshow(Gray_Pro, cmap='gray')
bx1 = plt.subplot(2, 2, 2)
bx1.set_title('gray segment result')
bx1.imshow(gray_out, cmap='gray')
# 三维时的贝叶斯决策
# ------------------------------------------------------------------------------------#
# 数据为三维时(彩色图像)，用核密度估计（非参数方法）求出两个类别条件概率pdf
# 用最大后验贝叶斯对彩色图像进行分割
RGB1_m = np.mean(RGB1, axis=0)
RGB2_m = np.mean(RGB2, axis=0)
RGB3_m = np.mean(RGB3, axis=0)
cov_sum1 = np.zeros((3, 3))
cov_sum2 = np.zeros((3, 3))
cov_sum3 = np.zeros((3, 3))
for i in range(len(RGB1)):
    cov_sum1 = cov_sum1 + np.dot((RGB1[i]-RGB1_m).reshape(3, 1), (RGB1[i]-RGB1_m).reshape(1, 3)) #平方差
for i in range(len(RGB2)):
    cov_sum2 = cov_sum2 + np.dot((RGB2[i]-RGB2_m).reshape(3, 1), (RGB2[i]-RGB2_m).reshape(1, 3))
for i in range(len(RGB3)):
    cov_sum3 = cov_sum3 + np.dot((RGB3[i]-RGB3_m).reshape(3, 1), (RGB3[i]-RGB3_m).reshape(1, 3))
RGB1_cov = cov_sum1/(len(RGB1)-1)                     # 无偏估计除以N-1
RGB2_cov = cov_sum2/(len(RGB2)-1)
RGB3_cov = cov_sum3/(len(RGB3)-1)
RGB_out = np.zeros_like(RGB_Pro)
for i in range(len(RGB_Pro)):
    for j in range(len(RGB_Pro[0])):
        if np.sum(RGB_Pro[i][j]) == 0:
            continue
            # 正态分布下的最大后验概率贝叶斯决策的分子比较
        elif max(P_pre1*multivariate_normal.pdf(RGB_Pro[i][j], RGB1_m, RGB1_cov),P_pre2*multivariate_normal.pdf(RGB_Pro[i][j], RGB2_m, RGB2_cov),P_pre3*multivariate_normal.pdf(RGB_Pro[i][j], RGB3_m, RGB3_cov)) == P_pre1*multivariate_normal.pdf(RGB_Pro[i][j], RGB1_m, RGB1_cov):
            RGB_out[i][j] = [255, 0, 0]
        elif max(P_pre1*multivariate_normal.pdf(RGB_Pro[i][j], RGB1_m, RGB1_cov),P_pre2*multivariate_normal.pdf(RGB_Pro[i][j], RGB2_m, RGB2_cov),P_pre3*multivariate_normal.pdf(RGB_Pro[i][j], RGB3_m, RGB3_cov)) == P_pre2*multivariate_normal.pdf(RGB_Pro[i][j], RGB2_m, RGB2_cov):
            RGB_out[i][j] = [0, 255, 0]
        else:
            RGB_out[i][j] = [0, 0, 0]
# 显示RGB ROI，与彩色分割结果
cx = plt.subplot(2, 2, 3)
cx.set_title('RGB nemo')
cx.imshow(RGB_Pro)
cx1 = plt.subplot(2, 2, 4)
cx1.set_title('RGB segment result')
cx1.imshow(RGB_out)
plt.show()

