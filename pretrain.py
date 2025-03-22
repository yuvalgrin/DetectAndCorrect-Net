import dataclasses

import numpy as np
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

    bins_clean = _plot_bars(clean_indices, loss_sorted_norm, noisy_indices)
    _plot_gaussians(bins_clean, gaus1, gaus2, loss_sorted_norm, threshold)

    plt.xlabel('Loss Value')
    plt.ylabel('Density')
    plt.title(f'Fitted Distributions and Threshold\nPredicted Noise Rate = {predicted_noise_rate:.3f}')
    plt.legend()
    plt.grid(True)
    hist_path = matrix_dir + '/' + 'noise_hist.png'
    plt.savefig(hist_path)


def _plot_gaussians(bins_clean, gaus1, gaus2, loss_sorted_norm, threshold):
    x = np.linspace(np.min(loss_sorted_norm), np.max(loss_sorted_norm), 1000)
    width = bins_clean[1] - bins_clean[0]
    plt.plot(x,
             gaus1.weight * width * norm.pdf(x, loc=gaus1.mean, scale=gaus1.std),
             'r-', lw=2, label=f'Normal Fit 1 [mean={gaus1.mean:.0f},std={gaus1.std:.0f}]')
    plt.plot(x,
             gaus2.weight * width * norm.pdf(x, loc=gaus2.mean, scale=gaus2.std),
             'g-', lw=2, label=f'Normal Fit 2 [mean={gaus2.mean:.0f},std={gaus2.std:.0f}]')
    plt.axvline(threshold, color='k', linestyle='dashed', linewidth=2, label=f'Threshold = {threshold:.2f}')


def _plot_bars(clean_indices, loss_sorted_norm, noisy_indices):
    hist_clean, bins_clean = np.histogram(loss_sorted_norm[clean_indices], bins=50)
    hist_noisy, bins_noisy = np.histogram(loss_sorted_norm[noisy_indices], bins=50)
    total_data_points = len(loss_sorted_norm)
    hist_clean = hist_clean / total_data_points
    hist_noisy = hist_noisy / total_data_points
    plt.bar(bins_clean[:-1], hist_clean, width=np.diff(bins_clean), alpha=0.6, color='green', label='Clean',
            align='edge')
    plt.bar(bins_noisy[:-1], hist_noisy, width=np.diff(bins_noisy), alpha=0.6, color='red', label='Noisy', align='edge')
    return bins_clean


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

    return transition_matrix
