# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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
"""DIPY-like models (a sandbox to trial them out before upstreaming to DIPY)."""
from __future__ import annotations

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from dipy.core.gradients import GradientTable
from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit


def gp_prediction(
    model: GaussianProcessRegressor,
    gtab: np.ndarray,
    mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicts one or more DWI orientations given a model.

    This function checks if the model is fitted and then extracts
    orientations and potentially b-values from the gtab. It predicts the mean
    and standard deviation of the DWI signal using the model.

    Parameters
    ----------
    model: :obj:`~sklearn.gaussian_process.GaussianProcessRegressor`
        A fitted GaussianProcessRegressor model.
    gtab: :obj:`~dipy.core.gradients.GradientTable`
        A gradient table containing diffusion encoding information.
    mask: :obj:`numpy.ndarray`
        A boolean mask indicating which voxels to use (optional).

    Returns
    -------
    :obj:`numpy.ndarray`
        A 3D or 4D array with the simulated gradient(s).

    """

    # Check it's fitted as they do in sklearn internally
    # https://github.com/scikit-learn/scikit-learn/blob/972e17fe1aa12d481b120ad4a3dc076bae736931/\
    # sklearn/gaussian_process/_gpr.py#L410C9-L410C42
    if not hasattr(model._gpr, "X_train_"):
        raise RuntimeError("Model is not yet fitted.")

    # Extract orientations from gtab, and highly likely, the b-value too.
    return model._gpr.predict(gtab, return_std=False)


def get_kernel(kernel_model: str) -> GaussianProcessRegressor.kernel:
    """
    Returns a Gaussian process kernel based on the provided string.

    Currently supports 'test' kernel which is a combination of DotProduct and WhiteKernel
    from scikit-learn. Raises a TypeError for unknown kernel models.

    Parameters
    ----------
    kernel_model: :obj:`str`
        The string representing the desired kernel model.

    Returns
    -------
    :obj:`GaussianProcessRegressor.kernel`
        A GaussianProcessRegressor kernel object.

    Raises
    ------
    TypeError: If the provided kernel_model is not supported.

    """

    if kernel_model == 'spherical':
        raise NotImplementedError("Spherical kernel is not currently implemented.")

    if kernel_model == 'exponential':
        raise NotImplementedError("Exponential kernel is not currently implemented.")

    if kernel_model == 'test':
        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

        return DotProduct() + WhiteKernel()

    raise TypeError(f"Unknown kernel '{kernel_model}'.")


class GaussianProcessModel(ReconstModel):
    """A Gaussian Process (GP) model to simulate single- and multi-shell DWI data."""

    __slots__ = (
        "kernel_model",
        "mask",
        "_gpr",
    )

    def __init__(
        self,
        gtab: GradientTable,
        kernel_model: str = "spherical",
        random_state: int = 0,
        *args,
        **kwargs,
    ):
        """A GP-based DWI model [Andersson15]_.

        Parameters
        ----------
        gtab : GradientTable class instance

        kernel_model : str
            Kernel model to calculate the GP's covariance matrix.

        References
        ----------
        .. [Andersson15] Jesper L.R. Andersson and Stamatios N. Sotiropoulos.
           Non-parametric representation and prediction of single- and multi-shell
           diffusion-weighted MRI data using Gaussian processes. NeuroImage, 122:166-176, 2015.
           doi:\
           `10.1016/j.neuroimage.2015.07.067 <https://doi.org/10.1016/j.neuroimage.2015.07.067>`__.

        """

        ReconstModel.__init__(self, gtab)
        self.kernel_model = kernel_model
        self.gtab = gtab
        self._gpr = GaussianProcessRegressor(
            kernel=get_kernel(self.kernel_model),
            random_state=random_state,
        )

    def fit(self, data, gtab=None, mask=None):
        """Fit method of the DTI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array, optional
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        """

        if mask is not None:
            data = data[mask[..., None]]
        else:
            data = np.reshape(data, (-1, data.shape[-1]))

        gtab = gtab if gtab is not None else self.gtab
        return GPFit(
            self._gpr.fit(gtab, data),
            gtab=gtab,
            mask=mask,
        )

    @multi_voxel_fit
    def multi_fit(self, data_thres, mask=None, **kwargs):
        return GPFit(self._gpr.fit(self.gtab, data_thres))

    def predict(self, gtab, mask=None, **kwargs):
        """Predict using the Gaussian process model of the DWI signal, where
        ``X`` is a diffusion-encoding gradient vector whose DWI data needs to be
        estimated.

        Parameters
        ----------
        gtab : :obj:`~dipy.core.gradients.GradientTable`
            One or more gradient orientations at which the GP will be evaluated.

        Returns
        -------
        :obj:`~numpy.ndarray` of shape (n_voxels, n_gradients)
            A 3D/4D array with the simulated voxels within the mask.

        """

        return gp_prediction(self._gpr, gtab, mask=mask)


class GPFit:
    def __init__(self, model, mask):
        self.model = model
        self.mask = mask

    def predict(self, gtab):
        return gp_prediction(self.model, gtab, mask=self.mask)
