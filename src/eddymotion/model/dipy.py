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

# import numpy as np


from sklearn.gaussian_process import GaussianProcessRegressor
from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit

def gp_prediction(model, gtab, mask=None):
    """Predict one or more DWI orientiations given a model."""

    # Check it's fitted as they do in sklearn internally
    # https://github.com/scikit-learn/scikit-learn/blob/972e17fe1aa12d481b120ad4a3dc076bae736931/\
    # sklearn/gaussian_process/_gpr.py#L410C9-L410C42
    if not hasattr(model._gpr, "X_train_"):
        raise RuntimeError("Model is not yet fitted.")

    # Extract orientations from gtab, and highly likely, the b-value too.
    y_mean, y_std = self._gpr.predict(gtab, return_std=True)

    # TODO reshape to be N voxels x G gradient directions (typically only one)

    return y_mean, y_std


class GaussianProcessModel(ReconstModel):
    """A Gaussian Process (GP) model to simulate single- and multi-shell DWI data."""

    __slots__ = (
        "kernel_model",
        "mask",
        "_gpr",
    )

    def __init__(self, gtab, kernel_model="spherical", mask=None, *args, **kwargs):
        """A GP-based DWI model [Andersson15]_.

        Parameters
        ----------
        gtab : GradientTable class instance

        kernel_model : str
            Kernel model to calculate the GP's covariance matrix.

        args, kwargs :
            arguments and key-word arguments passed to the fit_method.
            See dti.wls_fit_tensor, dti.ols_fit_tensor for details

        min_signal : float, optional
            The minimum signal value. Needs to be a strictly positive
            number. Default: minimal signal in the data provided to `fit`.

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
        self.mask = mask

    @multi_voxel_fit
    def fit(self, data, mask=None, random_state=0, **kwargs):
        """Fit method of the DTI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array, optional
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[:-1]

        """

        if self.kernel_model == 'spherical':
            raise NotImplementedError
        elif self.kernel_model == 'exponential':
            raise NotImplementedError
        elif self.kernel_model == 'test':
            from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

            kernel = DotProduct() + WhiteKernel()

        else:
            raise TypeError(f"Unknown kernel '{self.kernel_model}'.")

        self._gpr = GaussianProcessRegressor(kernel=kernel, random_state=random_state)

        if mask is not None:
            self.mask = mask
            data = data[mask]

        return GPFit(self._gpr.fit(self._gtab, data))

    def predict(self, gtab, **kwargs):
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

        return gp_prediction(self._gpr, gtab, mask=self.mask)


class GPFit:
    def __init__(self, model, mask):
        self.model = model
        self.mask = mask

    def predict(self, gtab):
        return gp_prediction(self.model, gtab, mask=self.mask)
