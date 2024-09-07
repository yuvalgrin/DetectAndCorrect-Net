import dataclasses
import pickle

import numpy as np
import wandb
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from matplotlib import pyplot as plt


@dataclasses.dataclass
class NormalFit:
    mean: float
    std: float
    weight: float


def plot_noise_hist(loss_sorted_norm, ind_1_sorted, gaus1, gaus2, threshold, predicted_noise_rate, matrix_dir, noise_or_not):
    """Plot the histogram of the normalized loss values and the fitted distributions"""
    sorted_noise_or_not = noise_or_not[ind_1_sorted]
    clean_indices = np.where(sorted_noise_or_not == 1)[0]
    noisy_indices = np.where(sorted_noise_or_not == 0)[0]
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 6))
    _, bins_clean, _ = plt.hist(loss_sorted_norm[clean_indices], bins=50, density=False, alpha=0.6, color='green', label='Clean')
    plt.hist(loss_sorted_norm[noisy_indices], bins=50, density=False, alpha=0.6, color='red', label='Noisy')
    x = np.linspace(np.min(loss_sorted_norm), np.max(loss_sorted_norm), 1000)
    bin_width = bins_clean[1] - bins_clean[0]
    plt.plot(x, gaus1.weight * len(loss_sorted_norm) * bin_width * norm.pdf(x, loc=gaus1.mean, scale=gaus1.std), 'r-', lw=2, label=f'Normal Fit 1 [mean={gaus1.mean:.0f},std={gaus1.std:.0f}]')
    plt.plot(x, gaus2.weight * len(loss_sorted_norm) * bin_width * norm.pdf(x, loc=gaus2.mean, scale=gaus2.std), 'b-', lw=2, label=f'Normal Fit 2 [mean={gaus2.mean:.0f},std={gaus2.std:.0f}]')
    plt.axvline(threshold, color='k', linestyle='dashed', linewidth=2, label=f'Threshold = {threshold:.2f}')
    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    plt.title(f'Fitted Distributions and Threshold\nPredicted Noise Rate = {predicted_noise_rate:.3f}')
    plt.legend()
    plt.grid(True)
    hist_path = matrix_dir + '/' + 'noise_hist.png'
    plt.savefig(hist_path)


def fit_gmm(loss_sorted_norm, reg_covar=1e-1):
    """Fit a Gaussian Mixture Model to the normalized loss values."""
    scaler = StandardScaler()
    loss_sorted_norm_scaled = scaler.fit_transform(loss_sorted_norm.reshape(-1, 1))
    gmm = GaussianMixture(n_components=2, random_state=0, reg_covar=reg_covar)
    gmm.fit(loss_sorted_norm_scaled)
    means_unscaled = scaler.inverse_transform(gmm.means_).flatten()
    stds_unscaled = np.sqrt(gmm.covariances_).flatten() * scaler.scale_

    gaus1 = NormalFit(means_unscaled[0], stds_unscaled[0], gmm.weights_[0])
    gaus2 = NormalFit(means_unscaled[1], stds_unscaled[1], gmm.weights_[1])
    x_range = np.linspace(loss_sorted_norm.min(), loss_sorted_norm.max(), 1000)
    x_range_scaled = scaler.transform(x_range.reshape(-1, 1))
    gmm_predict = gmm.predict(x_range_scaled)
    threshold_index = np.argmax(gmm_predict[:-1] != gmm_predict[1:])
    threshold = x_range[threshold_index]
    return gaus1, gaus2, threshold


def estimate_initial_t(data, counters, num_classes, is_noisy_sample):
    """Estimate the initial transition matrix."""
    transition_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    for loader, counter in zip(data, counters):
        img, label, idx = loader
        pred = counter.most_common(1)[0][0]
        if label != pred and is_noisy_sample[idx] == 1:
            transition_matrix[label, pred] += 1

    for i in range(num_classes):
        if transition_matrix[i].sum() > 0:
            transition_matrix[i] /= transition_matrix[i].sum()

    identity_matrix = np.eye(num_classes, dtype=np.float32)
    transition_matrix = (transition_matrix + identity_matrix) / 2

    return transition_matrix
