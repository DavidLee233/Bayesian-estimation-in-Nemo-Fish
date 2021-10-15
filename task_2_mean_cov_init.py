# 根据array_sample得到均值、协方差矩阵
import numpy as np
from scipy import io


def get_cov(pixels_rgb, pixels_means):
    cnt = len(pixels_rgb)
    cov_sum = np.zeros([3, 3])

    for i in range(cnt):
        cov_sum = cov_sum + np.dot((pixels_rgb[i] - pixels_means).reshape(3, 1),
                                   (pixels_rgb[i] - pixels_means).reshape(1, 3))
    cov = cov_sum / (cnt - 1)

    return cov


def mean_cov_times(means, covs, time):
    new_mean_1 = time * means[0]
    new_mean_2 = time * means[1]
    new_cov_1 = time * covs[0]
    new_cov_2 = time * covs[1]

    return new_mean_1, new_mean_2, new_cov_1, new_cov_2


def mean_cov_printf(names, means, covs, postfix):
    print('#################### ' + postfix)
    print(names[0] + '_mean_' + postfix + '=np.array(['
          + str(means[0][0]) + ', ' + str(means[0][1]) + ', ' + str(means[0][2]) + '])')
    print(names[1] + '_mean_' + postfix + '=np.array(['
          + str(means[1][0]) + ', ' + str(means[1][1]) + ', ' + str(means[1][2]) + '])')
    print(names[0] + '_cov_' + postfix + '=np.array([\n'
          + '[' + str(covs[0][0, 0]) + ', ' + str(covs[0][0, 1]) + ', ' + str(covs[0][0, 2]) + '],\n'
          + '[' + str(covs[0][1, 0]) + ', ' + str(covs[0][1, 1]) + ', ' + str(covs[0][1, 2]) + '],\n'
          + '[' + str(covs[0][2, 0]) + ', ' + str(covs[0][2, 1]) + ', ' + str(covs[0][2, 2]) + ']])')
    print(names[1] + '_cov_' + postfix + '=np.array([\n'
          + '[' + str(covs[1][0, 0]) + ', ' + str(covs[1][0, 1]) + ', ' + str(covs[1][0, 2]) + '],\n'
          + '[' + str(covs[1][1, 0]) + ', ' + str(covs[1][1, 1]) + ', ' + str(covs[1][1, 2]) + '],\n'
          + '[' + str(covs[1][2, 0]) + ', ' + str(covs[1][2, 1]) + ', ' + str(covs[1][2, 2]) + ']])')
    print()


sample_mat_file = 'array_sample.mat'
sample_mat_name = 'array_sample'


if __name__ == '__main__':
    pixels = io.loadmat(sample_mat_file)[sample_mat_name]

    pixels_gray = pixels[:, 0]
    pixels_rgb = pixels[:, 1:4]
    pixels_label = pixels[:, -1]

    white_label_bool = (pixels_label - 1).astype(np.bool8)
    red_label_bool = (pixels_label + 1).astype(np.bool8)

    pixels_rgb_white = pixels_rgb[white_label_bool]
    pixels_rgb_red = pixels_rgb[red_label_bool]

    white_mean = pixels_rgb_white.mean(axis=0)
    red_mean = pixels_rgb_red.mean(axis=0)

    white_cov = get_cov(pixels_rgb_white, white_mean)
    red_cov = get_cov(pixels_rgb_red, red_mean)

    for time in [0.3, 0.4, 0.5]:
        new_white_mean, new_red_mean, new_white_cov, new_red_cov = mean_cov_times((white_mean, red_mean), (white_cov, red_cov), time)
        mean_cov_printf(('white', 'red'), (new_white_mean, new_red_mean), (new_white_cov, new_red_cov), str(int(time * 10)))