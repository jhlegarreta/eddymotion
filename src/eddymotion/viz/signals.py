# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# © The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Visualizing signals and intermediate aspects of models."""

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


def plot_error(
    kfolds: list[int], mean: np.ndarray, std_dev: np.ndarray, xlabel: str, ylabel: str, title: str
) -> plt.Figure:
    """
    Plot the error and standard deviation.

    Parameters
    ----------
    kfolds : :obj:`list`
        Number of k-folds.
    mean : :obj:`~numpy.ndarray`
        Mean RMSE values.
    std_dev : :obj:`~numpy.ndarray`
        Standard deviation values.
    xlabel : :obj:`str`
        X-axis label.
    ylabel : :obj:`str`
        Y-axis label.
    title : :obj:`str`
        Plot title.

    Returns
    -------
    :obj:`~matplotlib.pyplot.Figure`
        Matplotlib figure object.

    """
    fig, ax = plt.subplots()
    ax.plot(kfolds, mean, c="orange")
    ax.fill_between(kfolds, mean - std_dev, mean + std_dev, alpha=0.5, color="orange")
    ax.scatter(kfolds, mean, c="orange")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(kfolds)
    ax.set_xticklabels(kfolds)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_estimation_carpet(gt_nii, gp_nii, gtab, suptitle, **kwargs):
    from nireports.reportlets.modality.dwi import nii_to_carpetplot_data
    from nireports.reportlets.nuisance import plot_carpet

    fig = plt.figure(layout="tight")
    gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    fig.suptitle(suptitle)

    divide_by_b0 = False
    gt_data, segments = nii_to_carpetplot_data(gt_nii, bvals=gtab.bvals, divide_by_b0=divide_by_b0)

    title = "Ground truth"
    plot_carpet(gt_data, segments, subplot=gs[0, :], title=title, **kwargs)

    gp_data, segments = nii_to_carpetplot_data(gp_nii, bvals=gtab.bvals, divide_by_b0=divide_by_b0)

    title = "Estimated (GP)"
    plot_carpet(gt_data, segments, subplot=gs[1, :], title=title, **kwargs)

    return fig


def plot_correlation(x, y, title):
    r = pearsonr(x, y)

    # Fit a linear curve and estimate its y-values and their error
    a, b = np.polyfit(x, y, deg=1)
    y_est = a * x + b
    y_err = x.std() * np.sqrt(1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    fig, ax = plt.subplots()
    ax.plot(x, y_est, "-", color="black", label=f"r = {r.correlation:.2f}")
    ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2, color="lightgray")
    ax.plot(x, y, marker="o", markersize="4", color="gray")

    ax.set_ylabel("Ground truth")
    ax.set_xlabel("Estimated")

    plt.title(title)
    plt.legend(loc="lower right")

    fig.tight_layout()

    return fig, r
