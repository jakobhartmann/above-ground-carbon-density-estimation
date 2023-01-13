import numpy as np
from scipy.stats import norm

def l1(bo_mean, ground_truth):
    return np.sum(np.abs(bo_mean - ground_truth))


def l2(bo_mean, ground_truth):
    return np.sum((bo_mean - ground_truth) ** 2)


def mse(bo_mean, ground_truth):
    return np.mean((bo_mean - ground_truth) ** 2)


def model_variance(bo_std):
    return np.mean(bo_std ** 2)


def mpdf(bo_mean, bo_std, ground_truth):
    return np.mean(norm.pdf(ground_truth, bo_mean, bo_std))


def kl(bo_mean, bo_var, ground_truth, ground_truth_var = 10 ** -8):
    return np.mean(np.log(bo_var / ground_truth_var) + ((ground_truth_var + (ground_truth - bo_mean) ** 2) / (2 * bo_var)) - 0.5)


def psnr(bo_mean, ground_truth):
    max_val = max([np.max(bo_mean), np.max(ground_truth)])
    return 10 * np.log10((max_val ** 2) / mse(bo_mean, ground_truth))


def ssim(bo_mean, ground_truth):
    # Dynamic range (of the pixel values)
    min_val = min([np.min(bo_mean), np.min(ground_truth)])
    max_val = max([np.max(bo_mean), np.max(ground_truth)])
    L = max_val / min_val # TODO Check dynamic range calculation

    # Constants for avoiding instability
    k1 = 0.01 # default value
    k2 = 0.03 # default value
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    c3 = c2 / 2

    # Weights
    alpha = 1
    beta = 1
    gamma = 1

    # Means
    mu_x = np.mean(bo_mean)
    mu_y = np.mean(ground_truth)

    # Variances and standard deviations
    var_x = np.var(bo_mean)
    std_x = np.sqrt(var_x)
    var_y = np.var(ground_truth)
    std_y = np.sqrt(var_y)

    # Covariance
    cov = np.cov(bo_mean.flatten(), ground_truth.flatten())
    cov_x_y = cov[0][1]

    # Luminance
    l = (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
    # Contrast
    c = (2 * std_x * std_y + c2) / (var_x + var_y + c2)
    # Structure
    s = (cov_x_y + c3) / (std_x * std_y + c3)

    # SSIM
    result = (l ** alpha) * (c ** beta) * (s ** gamma)

    return result

def calc_metrics(mu_plot, std_plot, ground_truth_reshaped, mu_unseen, std_unseen, ground_truth_unseen):
    L1 = l1(mu_plot, ground_truth_reshaped)
    L2 = l2(mu_plot, ground_truth_reshaped)
    MSE = mse(mu_plot, ground_truth_reshaped)
    PSNR = psnr(mu_plot, ground_truth_reshaped)
    SSIM = ssim(mu_plot, ground_truth_reshaped)
    MPDF_unseen = mpdf(mu_unseen, std_unseen, ground_truth_unseen)
    MPDF_all = mpdf(mu_plot, std_plot, ground_truth_reshaped)
    KL = kl(ground_truth_unseen, 10 ** -8, mu_unseen, std_unseen ** 2)
    ModelVariance_unseen = model_variance(std_unseen)
    ModelVariance_all = model_variance(std_plot)
    
    return L1, L2, MSE, PSNR, SSIM, MPDF_unseen, MPDF_all, KL, ModelVariance_unseen, ModelVariance_all
