# 基于无标签样本数据对309.bmp灰度图进行处理
import os

from utils import *


# 迭代开始
def img_rgb_segmentation_using_em(sample, img_and_mask, save_info, epochs, priors_init, means_init, covs_init):
    color_1_prior = priors_init[0]
    color_2_prior = priors_init[1]

    color_1_mean = means_init[0]
    color_2_mean = means_init[1]

    color_1_cov = covs_init[0]
    color_2_cov = covs_init[1]

    # EM算法
    labels = np.zeros((sample.shape[0], 2))
    for epoch in range(epochs):
        print('EM: Epoch_' + str(epoch + 1) + ' starting...')
        ### 第一步：E过程
        color_1_val = color_1_prior * multivariate_normal.pdf(sample, color_1_mean, color_1_cov)
        color_2_val = color_2_prior * multivariate_normal.pdf(sample, color_2_mean, color_2_cov)

        color_1_label = color_1_val / (color_1_val + color_2_val)
        color_2_label = 1 - color_1_label

        labels[:, 0], labels[:, 1] = color_1_label, color_2_label

        color_1_rgb = sample * np.array(color_1_label)[:, np.newaxis]
        color_2_rgb = sample * np.array(color_2_label)[:, np.newaxis]

        ### 第二步：M过程
        # 更新先验概率
        color_1_cnt = labels[:, 0].sum()
        color_2_cnt = labels[:, 1].sum()
        color_1_prior = color_1_cnt / (color_1_cnt + color_2_cnt)
        color_2_prior = 1 - color_1_prior

        # 更新均值
        color_1_mean = color_1_rgb.sum(axis=0) / color_1_cnt
        color_2_mean = color_2_rgb.sum(axis=0) / color_2_cnt

        # 更新协方差矩阵
        color_1_cov_sum = np.zeros([3, 3])
        color_2_cov_sum = np.zeros([3, 3])
        for i in range(len(sample)):
            color_1_cov_sum = color_1_cov_sum + np.dot((sample[i] - color_1_mean).reshape(3, 1),
                                                       (sample[i] - color_1_mean).reshape(1, 3)) * labels[i, 0]
            color_2_cov_sum = color_2_cov_sum + np.dot((sample[i] - color_2_mean).reshape(3, 1),
                                                       (sample[i] - color_2_mean).reshape(1, 3)) * labels[i, 1]

        color_1_cov = color_1_cov_sum / (color_1_cnt - 1)  # 无偏估计除以N-1
        color_2_cov = color_2_cov_sum / (color_2_cnt - 1)

        ### 第三步：使用当前参数处理图像并保存
        img_rgb_processed = segment_img_rgb(img_and_mask[0], img_and_mask[1], (color_1_prior, color_2_prior),
                                            (color_1_mean, color_2_mean), (color_1_cov, color_2_cov))
        img_rgb_save(save_info[0], save_info[1] + '_' + 'Epoch-' + str(epoch + 1), save_info[2], img_rgb_processed)


# 超参数
img_dir = 'imgs'
img_to_process = '309.bmp'

sample_mat_file = 'array_sample.mat'
sample_mat_name = 'array_sample'

mask_mat_file = 'Mask.mat'
mask_mat_name = 'Mask'

out_dir = 'out/task_2'

epochs = 30
# 初始化先验概率
white_prior = 0.5
red_prior = 1 - white_prior

#################### 3
white_mean_3=np.array([0.2251504450464373, 0.21357294891640513, 0.2276335139318895])
red_mean_3=np.array([0.19584257107098413, 0.10833385481852523, 0.0487345789379579])
white_cov_3=np.array([
[0.005088976593381132, 0.005500585698764952, 0.006550825540331084],
[0.005500585698764952, 0.008195465755726118, 0.011614901213391232],
[0.006550825540331084, 0.011614901213391232, 0.019333166661934575]])
red_cov_3=np.array([
[0.008783331903143114, 0.0037678129574543256, 0.0008150420888208211],
[0.0037678129574543256, 0.0027075747006391248, 0.001978079735407847],
[0.0008150420888208211, 0.001978079735407847, 0.003333426611935179]])

#################### 4
white_mean_4=np.array([0.3002005933952498, 0.28476393188854016, 0.30351135190918604])
red_mean_4=np.array([0.2611234280946455, 0.14444513975803364, 0.06497943858394388])
white_cov_4=np.array([
[0.006785302124508176, 0.007334114265019937, 0.00873443405377478],
[0.007334114265019937, 0.01092728767430149, 0.015486534951188312],
[0.00873443405377478, 0.015486534951188312, 0.0257775555492461]])
red_cov_4=np.array([
[0.01171110920419082, 0.005023750609939101, 0.0010867227850944283],
[0.005023750609939101, 0.003610099600852167, 0.0026374396472104624],
[0.0010867227850944283, 0.0026374396472104624, 0.004444568815913572]])

#################### 5
white_mean_5=np.array([0.3752507417440622, 0.3559549148606752, 0.3793891898864825])
red_mean_5=np.array([0.3264042851183069, 0.18055642469754205, 0.08122429822992984])
white_cov_5=np.array([
[0.00848162765563522, 0.00916764283127492, 0.010918042567218475],
[0.00916764283127492, 0.013659109592876863, 0.019358168688985388],
[0.010918042567218475, 0.019358168688985388, 0.032221944436557626]])
red_cov_5=np.array([
[0.014638886505238523, 0.006279688262423876, 0.0013584034813680353],
[0.006279688262423876, 0.004512624501065208, 0.003296799559013078],
[0.0013584034813680353, 0.003296799559013078, 0.005555711019891965]])

#################### 10
white_mean_10=np.array([0.7505014834881244, 0.7119098297213504, 0.758778379772965])
red_mean_10=np.array([0.6528085702366138, 0.3611128493950841, 0.16244859645985968])
white_cov_10=np.array([
[0.01696325531127044, 0.01833528566254984, 0.02183608513443695],
[0.01833528566254984, 0.027318219185753726, 0.038716337377970776],
[0.02183608513443695, 0.038716337377970776, 0.06444388887311525]])
red_cov_10=np.array([
[0.029277773010477046, 0.012559376524847753, 0.0027168069627360705],
[0.012559376524847753, 0.009025249002130416, 0.006593599118026156],
[0.0027168069627360705, 0.006593599118026156, 0.01111142203978393]])

#################### 13
white_mean_13=np.array([0.9756519285345617, 0.9254827786377555, 0.9864118937048546])
red_mean_13=np.array([0.8486511413075979, 0.46944670421360934, 0.21118317539781759])
white_cov_13=np.array([
[0.022052231904651574, 0.023835871361314796, 0.028386910674768034],
[0.023835871361314796, 0.03551368494147984, 0.05033123859136201],
[0.028386910674768034, 0.05033123859136201, 0.08377705553504983]])
red_cov_13=np.array([
[0.038061104913620164, 0.01632718948230208, 0.0035318490515568917],
[0.01632718948230208, 0.011732823702769542, 0.008571678853434003],
[0.0035318490515568917, 0.008571678853434003, 0.01444484865171911]])


if __name__ == '__main__':
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    _, rgb_arr, _ = get_fish_sample(sample_mat_file, sample_mat_name)
    mask = get_mask(mask_mat_file, mask_mat_name)
    img_masked = get_img_rgb_masked(img_dir + '/' + img_to_process, mask)
    # EM算法
    img_rgb_segmentation_using_em(rgb_arr, (img_dir + '/' + img_to_process, mask), (out_dir, 'EM', img_to_process),
                                  epochs, (white_prior, red_prior), (white_mean_5, red_mean_5), (white_cov_5, red_cov_5))