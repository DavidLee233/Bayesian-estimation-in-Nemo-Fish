# 基于样本数据对311.bmp、313.bmp、315.bmp、317.bmp灰度图进行处理
import os

from utils import *


# 超参数
img_dir = 'imgs'
img_sample = '309.bmp'
imgs_to_process = ['311.bmp', '313.bmp', '315.bmp', '317.bmp']
height_range = [0, 170]
width_range = [50, 250]

sample_mat_file = 'array_sample.mat'
sample_mat_name = 'array_sample'

mask_file = 'Mask.mat'
mask_name = 'Mask'

out_dir = 'out/task_1-2'

weight_to_get_mask = 1.2
weight_to_classify_colors = 0.2


if __name__ == '__main__':
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    ### 生成Mask
    # 由样本计算前景（Nemo鱼）和背景假设分布模型的参数
    fg_pixels, bg_pixels = get_whole_sample(img_dir + '/' + img_sample, mask_file, mask_name)
    fg_prior, bg_prior = get_prior(fg_pixels, bg_pixels)
    fg_mean_rgb, bg_mean_rgb, fg_cov_rgb, bg_cov_rgb = get_mean_cov(fg_pixels, bg_pixels)
    # 通过多元高斯分布生成Mask
    for img in imgs_to_process:
        generate_mask(out_dir, img_dir, img, height_range, width_range,
                      (fg_prior, bg_prior), (fg_mean_rgb, bg_mean_rgb), (fg_cov_rgb, bg_cov_rgb), weight_to_get_mask)
    ### 由样本计算假设分布模型的参数
    gray_arr, rgb_arr, label_arr = get_fish_sample(sample_mat_file, sample_mat_name)
    # 基于灰度图
    white_pixels_gray, red_pixels_gray = get_two_color_pixels(gray_arr, label_arr)
    white_prior, red_prior = get_prior(white_pixels_gray, red_pixels_gray)
    white_mean_gray, red_mean_gray, white_std_gray, red_std_gray = get_mean_std(white_pixels_gray, red_pixels_gray)

    # 处理图片并保存
    for img in imgs_to_process:
        mask = get_mask(out_dir + '/' + 'Mask_of_' + img + '.mat', mask_name)
        img_gray_processed = segment_img_gray(img_dir + '/' + img, mask, (white_prior, red_prior),
                                              (white_mean_gray, red_mean_gray), (white_std_gray, red_std_gray),
                                              weight_to_classify_colors)
        img_gray_save(out_dir, 'basedOnGray', img, img_gray_processed)