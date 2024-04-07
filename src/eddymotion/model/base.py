# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""A factory class that adapts DIPY's dMRI models."""

import warnings

import numpy as np
from dipy.core.gradients import gradient_table
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor


def _exec_fit(model, data, chunk=None):
    retval = model.fit(data)
    return retval, chunk


def _exec_predict(model, gradient, chunk=None, **kwargs):
    """Propagate model parameters and call predict."""
    return np.squeeze(model.predict(gradient, S0=kwargs.pop("S0", None))), chunk


class ModelFactory:
    """A factory for instantiating diffusion models."""

    @staticmethod
    def init(model="DTI", **kwargs):
        """
        Instantiate a diffusion model.

        Parameters
        ----------
        model : :obj:`str`
            Diffusion model.
            Options: ``"DTI"``, ``"DKI"``, ``"S0"``, ``"AverageDW"``

        Return
        ------
        model : :obj:`~dipy.reconst.ReconstModel`
            A model object compliant with DIPY's interface.

        """
        if model.lower() in ("s0", "b0"):
            return TrivialB0Model(S0=kwargs.pop("S0"))

        if model.lower() in ("avg", "average", "mean"):
            return AverageDWModel(**kwargs)

        if model.lower() in ("dti", "dki", "pet"):
            Model = globals()[f"{model.upper()}Model"]
            return Model(**kwargs)

        raise NotImplementedError(f"Unsupported model <{model}>.")


class BaseModel:
    """
    Defines the interface and default methods.

    Implements the interface of :obj:`dipy.reconst.base.ReconstModel`.
    Instead of inheriting from the abstract base, this implementation
    follows type adaptation principles, as it is easier to maintain
    and to read (see https://www.youtube.com/watch?v=3MNVP9-hglc).

    """

    __slots__ = (
        "_model",
        "_mask",
        "_S0",
        "_b_max",
        "_models",
        "_datashape",
    )
    _modelargs = ()

    def __init__(self, gtab, S0=None, mask=None, b_max=None, **kwargs):
        """Base initialization."""

        # Setup B0 map
        self._S0 = None
        if S0 is not None:
            self._S0 = np.clip(
                S0.astype("float32") / S0.max(),
                a_min=1e-5,
                a_max=1.0,
            )

        # Setup brain mask
        self._mask = mask
        if mask is None and S0 is not None:
            self._mask = self._S0 > np.percentile(self._S0, 35)

        # Cap b-values, if requested
        self._b_max = None
        if b_max and b_max > 1000:
            # Saturate b-values at b_max, since signal stops dropping
            gtab[-1, gtab[-1] > b_max] = b_max
            # A possibly good alternative is completely remove very high b-values
            # bval_mask = gtab[-1] < b_max
            # data = data[..., bval_mask]
            # gtab = gtab[:, bval_mask]
            self._b_max = b_max

        kwargs = {k: v for k, v in kwargs.items() if k in self._modelargs}

        model_str = getattr(self, "_model_class", None)
        if not model_str:
            raise TypeError("No model defined")

        # ToDo
        # Use lazy loading ?
        from importlib import import_module

        module_name, class_name = model_str.rsplit(".", 1)
        self._model = getattr(import_module(module_name), class_name)(_rasb2dipy(gtab), **kwargs)

        self._datashape = None
        self._models = None

    def fit(self, data, n_jobs=None, **kwargs):
        """Fit the model chunk-by-chunk asynchronously"""
        n_jobs = n_jobs or 1

        self._datashape = data.shape

        # Select voxels within mask or just unravel 3D if no mask
        data = (
            data[self._mask, ...] if self._mask is not None else data.reshape(-1, data.shape[-1])
        )

        # One single CPU - linear execution (full model)
        if n_jobs == 1:
            self._model, _ = _exec_fit(self._model, data)
            return

        # Split data into chunks of group of slices
        data_chunks = np.array_split(data, n_jobs)

        self._models = [None] * n_jobs

        # Parallelize process with joblib
        with Parallel(n_jobs=n_jobs) as executor:
            results = executor(
                delayed(_exec_fit)(self._model, dchunk, i) for i, dchunk in enumerate(data_chunks)
            )
        for submodel, index in results:
            self._models[index] = submodel

        self._model = None  # Preempt further actions on the model

    def predict(self, gradient, **kwargs):
        """Predict asynchronously chunk-by-chunk the diffusion signal."""
        if self._b_max is not None:
            gradient[-1] = min(gradient[-1], self._b_max)

        gradient = _rasb2dipy(gradient)

        S0 = None
        if self._S0 is not None:
            S0 = (
                self._S0[self._mask, ...]
                if self._mask is not None
                else self._S0.reshape(-1, self._S0.shape[-1])
            )

        n_models = len(self._models) if self._model is None and self._models else 1

        if n_models == 1:
            predicted, _ = _exec_predict(self._model, gradient, S0=S0, **kwargs)
        else:
            S0 = np.array_split(S0, n_models) if S0 is not None else [None] * n_models

            predicted = [None] * n_models

            # Parallelize process with joblib
            with Parallel(n_jobs=n_models) as executor:
                results = executor(
                    delayed(_exec_predict)(model, gradient, S0=S0[i], chunk=i, **kwargs)
                    for i, model in enumerate(self._models)
                )
            for subprediction, index in results:
                predicted[index] = subprediction

            predicted = np.hstack(predicted)

        if self._mask is not None:
            retval = np.zeros_like(self._mask, dtype="float32")
            retval[self._mask, ...] = predicted
        else:
            retval = predicted.reshape(self._datashape[:-1])

        return retval


class TrivialB0Model:
    """A trivial model that returns a *b=0* map always."""

    __slots__ = ("_S0",)

    def __init__(self, S0=None, **kwargs):
        """Implement object initialization."""
        if S0 is None:
            raise ValueError("S0 must be provided")

        self._S0 = S0

    def fit(self, *args, **kwargs):
        """Do nothing."""

    def predict(self, gradient, **kwargs):
        """Return the *b=0* map."""
        return self._S0


class AverageDWModel:
    """A trivial model that returns an average map."""

    __slots__ = ("_data", "_th_low", "_th_high", "_bias", "_stat")

    def __init__(self, **kwargs):
        r"""
        Implement object initialization.

        Parameters
        ----------
        gtab : :obj:`~numpy.ndarray`
            An :math:`N \times 4` table, where rows (*N*) are diffusion gradients and
            columns are b-vector components and corresponding b-value, respectively.
        th_low : :obj:`~numbers.Number`
            A lower bound for the b-value corresponding to the diffusion weighted images
            that will be averaged.
        th_high : :obj:`~numbers.Number`
            An upper bound for the b-value corresponding to the diffusion weighted images
            that will be averaged.
        bias : :obj:`bool`
            Whether the overall distribution of each diffusion weighted image will be
            standardized and centered around the global 75th percentile.
        stat : :obj:`str`
            Whether the summary statistic to apply is ``"mean"`` or ``"median"``.

        """
        self._th_low = kwargs.get("th_low", 50)
        self._th_high = kwargs.get("th_high", 10000)
        self._bias = kwargs.get("bias", True)
        self._stat = kwargs.get("stat", "median")
        self._data = None

    def fit(self, data, **kwargs):
        """Calculate the average."""
        gtab = kwargs.pop("gtab", None)
        # Select the interval of b-values for which DWIs will be averaged
        b_mask = (
            ((gtab[3] >= self._th_low) & (gtab[3] <= self._th_high))
            if gtab is not None
            else np.ones((data.shape[-1],), dtype=bool)
        )
        shells = data[..., b_mask]

        # Regress out global signal differences
        if self._bias:
            centers = np.median(shells, axis=(0, 1, 2))
            reference = np.percentile(centers[centers >= 1.0], 75)
            centers[centers < 1.0] = reference
            drift = reference / centers
            shells = shells * drift

        # Select the summary statistic
        avg_func = np.median if self._stat == "median" else np.mean
        # Calculate the average
        self._data = avg_func(shells, axis=-1)

    def predict(self, gradient, **kwargs):
        """Return the average map."""
        return self._data


class PETModel:
    """A PET imaging realignment model based on B-Spline approximation."""

    __slots__ = ("_t", "_x", "_xlim", "_order", "_coeff", "_mask", "_shape", "_n_ctrl")

    def __init__(self, timepoints=None, xlim=None, n_ctrl=None, mask=None, order=3, **kwargs):
        """
        Create the B-Spline interpolating matrix.

        Parameters:
        -----------
        timepoints : :obj:`list`
            The timing (in sec) of each PET volume.
            E.g., ``[15.,   45.,   75.,  105.,  135.,  165.,  210.,  270.,  330.,
            420.,  540.,  750., 1050., 1350., 1650., 1950., 2250., 2550.]``

        n_ctrl : :obj:`int`
            Number of B-Spline control points. If `None`, then one control point every
            six timepoints will be used. The less control points, the smoother is the
            model.

        """
        if timepoints is None or xlim is None:
            raise TypeError("timepoints must be provided in initialization")

        self._order = order
        self._mask = mask

        self._x = np.array(timepoints, dtype="float32")
        self._xlim = xlim

        if self._x[0] < 1e-2:
            raise ValueError("First frame midpoint should not be zero or negative")
        if self._x[-1] > (self._xlim - 1e-2):
            raise ValueError("Last frame midpoint should not be equal or greater than duration")

        # Calculate index coordinates in the B-Spline grid
        self._n_ctrl = n_ctrl or (len(timepoints) // 4) + 1

        # B-Spline knots
        self._t = np.arange(-3, float(self._n_ctrl) + 4, dtype="float32")

        self._shape = None
        self._coeff = None

    def fit(self, data, *args, **kwargs):
        """Fit the model."""
        from scipy.interpolate import BSpline
        from scipy.sparse.linalg import cg

        n_jobs = kwargs.pop("n_jobs", None) or 1

        timepoints = kwargs.get("timepoints", None) or self._x
        x = (np.array(timepoints, dtype="float32") / self._xlim) * self._n_ctrl

        self._shape = data.shape[:3]

        # Convert data into V (voxels) x T (timepoints)
        data = data.reshape((-1, data.shape[-1])) if self._mask is None else data[self._mask]

        # A.shape = (T, K - 4); T= n. timepoints, K= n. knots (with padding)
        A = BSpline.design_matrix(x, self._t, k=self._order)
        AT = A.T
        ATdotA = AT @ A

        # One single CPU - linear execution (full model)
        if n_jobs == 1:
            self._coeff = np.array([cg(ATdotA, AT @ v)[0] for v in data])
            return

        # Parallelize process with joblib
        with Parallel(n_jobs=n_jobs) as executor:
            results = executor(delayed(cg)(ATdotA, AT @ v) for v in data)

        self._coeff = np.array([r[0] for r in results])

    def predict(self, timepoint, **kwargs):
        """Return the *b=0* map."""
        from scipy.interpolate import BSpline

        # Project sample timing into B-Spline coordinates
        x = (timepoint / self._xlim) * self._n_ctrl
        A = BSpline.design_matrix(x, self._t, k=self._order)

        # A is 1 (num. timepoints) x C (num. coeff)
        # self._coeff is V (num. voxels) x K - 4
        predicted = np.squeeze(A @ self._coeff.T)

        if self._mask is None:
            return predicted.reshape(self._shape)

        retval = np.zeros(self._shape, dtype="float32")
        retval[self._mask] = predicted
        return retval


class DTIModel(BaseModel):
    """A wrapper of :obj:`dipy.reconst.dti.TensorModel`."""

    _modelargs = (
        "min_signal",
        "return_S0_hat",
        "fit_method",
        "weighting",
        "sigma",
        "jac",
    )
    _model_class = "dipy.reconst.dti.TensorModel"


class DKIModel(BaseModel):
    """A wrapper of :obj:`dipy.reconst.dki.DiffusionKurtosisModel`."""

    _modelargs = DTIModel._modelargs
    _model_class = "dipy.reconst.dki.DiffusionKurtosisModel"


class GaussianProcessModel:
    """A Gaussian Process model based on [Andersson16a]_ (fig 1).
    DWIs need to be transformed to a single ref space (fig 2 [Andersson16b]_ ?)

    Definitions:
    s: reference/undistorted space: used to denote the space or any image in
    that space
    f: observed/distorted image: used to denote any image in acquisition space
    a: acquisition parameters: PE‐direction and bandwidth in PE‐direction
    r: rigid body (subject movement) parameters
    \beta: Eddy current parameters
    e(\beta): Eddy current‐induced off resonance field (Hz)
    h: Susceptibility induced off‐resonance field (Hz)

    (fig 1) and algorithm:
    1. Input: N DWI volumes f_{i} with acq parameters a_{i}; susceptibility
    field h
    2. Initialize: set all beta_{i} and r_i{i} = 0
    3. Compute for M iterations
     - Load GP prediction maker
     - For all i in N (DWIs)
       - Compute \\hat{s}_{i} (f_{i}, h, \beta_{i}, r_{i}, a_{i}) eqs 2 and 4
       - Load \\hat{s}_{i} (f_{i}, h, \beta_{i}, r_{i}, a_{i}) as training
         data for GP
     - Estimate hyperparameters for the GP used to predict the signal shape
       for every voxel
     - Update EC and movement parameters
     - For all i in N (DWIs)
       - Draw a prediction s_{i} from the GP
       - Compute \\hat{f}_{i} (s_{i}, h, \beta_{i}, r_{i}, a_{i})
       - Use \\hat{f}_{i} - f_{i} to update \beta_{i} and r_{i} (eq 6)

    a: direction of the PE and the total readout time (here defined as the time
    between the acquisition of the center of the first and last echoes).
    Internally, a is divided into a = [p t] where p is a unity length 1 x 3
    vector defining the PE direction (such that for example [1 0 0], [−1 0 0],
    [0 1 0] and [0 −1 0] denote R → L, L → R, P → A and A  P PE direction
    respectively) and where t denotes the readout time (in seconds).
    r: 1x6 vector: 3 translations, 3 rotations
    \beta: four for linear; ten for quadratic, twenty for cubic.
    h: assumed to be in the same space as the first b = 0 image supplied to
    eddy, which will be automatically fulfilled if it was estimated by topup
    and that same b = 0 image was the first of those supplied to topup. Hence,
    it can be said to help define the reference/undistorted space as the first
    b = 0 image after distortion correction by h.

    See Appendix A for further details.

    Add the outlier detection part in [Andersson16b]?

    References
    ----------
    .. [Andersson16a] J. L. R. Andersson. et al., An integrated approach to
    correction for off-resonance effects and subject movement in diffusion MR
    imaging, NeuroImage 125 (2016) 1063–1078
    .. [Andersson16b] J. L. R. Andersson. et al., Incorporating outlier
    detection and replacement into a non-parametric framework for movement and
    distortion correction of diffusion MR images, NeuroImage 141 (2016) 556–572
    """

    __slots__ = (
        "_dwi",
        "_a",
        "_h",
        "_kernel",
        "_num_iterations",
        "_betas",
        "_r",
        "_gpr",
        "_model",
    )

    def __init__(self, dwi, a, h, kernel, num_iterations=5, **kwargs):
        """Implement object initialization."""

        self._dwi = dwi  # The HDF5 file object: avoid having the entire 4D volume in memory
        self._a = a
        self._h = h
        self._num_iterations = num_iterations

        # Initialize
        self._betas = 0
        self._r = 0

        # ToDo
        # Build the GP kernel here or in fit ?
        self._gpr = None
        # Does the kernel depend on which data we use as the training data (i.e.
        # varies with the index we choose to predict)?
        self._kernel = kernel
        self._model = None

    def fit(self, X, y, *args, **kwargs):
        """The x are our gradient directions; the observations are our diffusion
        volumes.
        X: array-like of shape (n_samples, n_features), n_samples being
        the number of gradients, and the n_features the number of shells ?
        Or n_samples being the number of voxels in the DWI volume, and n_features
        being 3 (bvec coordinates)
        y: _array-like of shape (n_samples,) or (n_samples, n_targets)"""

        self._gpr = GaussianProcessRegressor(kernel=self._kernel, random_state=0)
        self._gpr.fit(X, y)

    def predict(self, X, **kwargs):
        """Return the Gaussian Process prediction according to [Andersson16]_
        where X is a gradient direction."""
        # ToDo
        # Call self._gprlog_marginal_likelihood for eq. 12 in Andersson 15 ?
        y_mean, y_std = self._gpr.predict(X, return_std=True)
        return y_mean, y_std


def _rasb2dipy(gradient):
    gradient = np.asanyarray(gradient)
    if gradient.ndim == 1:
        if gradient.size != 4:
            raise ValueError("Missing gradient information.")
        gradient = gradient[..., np.newaxis]

    if gradient.shape[0] != 4:
        gradient = gradient.T
    elif gradient.shape == (4, 4):
        print("Warning: make sure gradient information is not transposed!")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        retval = gradient_table(gradient[3, :], gradient[:3, :].T)
    return retval
