# 基于样本数据对309.bmp灰度图、RGB图进行处理
import os
from utils import *
# 超参数
img_dir = 'imgs'
img_to_process = '309.bmp'

sample_mat_file = 'array_sample.mat'
sample_mat_name = 'array_sample'

mask_file = 'Mask.mat'
mask_name = 'Mask'

weight_gray = 0.2
weight_rgb = 0.2
out_dir = 'out/task_1-1'


if __name__ == '__main__':
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    ### 由样本计算假设分布模型的参数
    gray_arr, rgb_arr, label_arr = get_fish_sample(sample_mat_file, sample_mat_name)
    # 基于灰度图
    white_pixels_gray, red_pixels_gray = get_two_color_pixels(gray_arr, label_arr)
    white_prior, red_prior = get_prior(white_pixels_gray, red_pixels_gray)
    white_mean_gray, red_mean_gray, white_std_gray, red_std_gray = get_mean_std(white_pixels_gray, red_pixels_gray)
    # 基于RGB
    white_pixels_rgb, red_pixels_rgb = get_two_color_pixels(rgb_arr, label_arr)
    white_mean_rgb, red_mean_rgb, white_cov_rgb, red_cov_rgb = get_mean_cov(white_pixels_rgb, red_pixels_rgb)

    ### 处理图片并保存
    mask = get_mask(mask_file, mask_name)
    # 基于灰度图
    img_gray_processed = segment_img_gray(img_dir + '/' + img_to_process, mask, (white_prior, red_prior),
                                          (white_mean_gray, red_mean_gray), (white_std_gray, red_std_gray), weight_gray)
    img_gray_save(out_dir, 'basedOnGray', img_to_process, img_gray_processed)
    # 基于RGB
    img_rgb_processed = segment_img_rgb(img_dir + '/' + img_to_process, mask, (white_prior, red_prior),
                                        (white_mean_rgb, red_mean_rgb), (white_cov_rgb, red_cov_rgb), weight_rgb)
    img_rgb_save(out_dir, 'basedOnRGB', img_to_process, img_rgb_processed)