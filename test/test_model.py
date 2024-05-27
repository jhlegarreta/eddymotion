# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
"""Unit tests exercising models."""

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from eddymotion import model
from eddymotion.data.dmri import DWI
from eddymotion.data.splitting import lovo_split


def test_trivial_model():
    """Check the implementation of the trivial B0 model."""

    # Should not allow initialization without a B0
    with pytest.raises(ValueError):
        model.TrivialB0Model(gtab=np.eye(4))

    _S0 = np.random.normal(size=(10, 10, 10))

    tmodel = model.TrivialB0Model(gtab=np.eye(4), S0=_S0)

    assert tmodel.fit() is None

    assert np.all(_S0 == tmodel.predict((1, 0, 0)))


def test_average_model():
    """Check the implementation of the average DW model."""

    data = np.ones((100, 100, 100, 6), dtype=float)

    gtab = np.array(
        [
            [0, 0, 0, 0],
            [-0.31, 0.933, 0.785, 25],
            [0.25, 0.565, 0.21, 500],
            [-0.861, -0.464, 0.564, 1000],
            [0.307, -0.766, 0.677, 1000],
            [0.736, 0.013, 0.774, 1300],
        ]
    )

    data *= gtab[:, -1]

    tmodel_mean = model.AverageDWModel(gtab=gtab, bias=False, stat="mean")
    tmodel_median = model.AverageDWModel(gtab=gtab, bias=False, stat="median")
    tmodel_1000 = model.AverageDWModel(gtab=gtab, bias=False, th_high=1000, th_low=900)
    tmodel_2000 = model.AverageDWModel(
        gtab=gtab,
        bias=False,
        th_high=2000,
        th_low=900,
        stat="mean",
    )

    # Verify that fit function returns nothing
    assert tmodel_mean.fit(data[..., 1:], gtab=gtab[1:].T) is None

    tmodel_median.fit(data[..., 1:], gtab=gtab[1:].T)
    tmodel_1000.fit(data[..., 1:], gtab=gtab[1:].T)
    tmodel_2000.fit(data[..., 1:], gtab=gtab[1:].T)

    # Verify that the right statistics is applied and that the model discard b-values < 50
    assert np.all(tmodel_mean.predict([0, 0, 0]) == 950)
    assert np.all(tmodel_median.predict([0, 0, 0]) == 1000)

    # Verify that the threshold for b-value selection works as expected
    assert np.all(tmodel_1000.predict([0, 0, 0]) == 1000)
    assert np.all(tmodel_2000.predict([0, 0, 0]) == 1100)


def test_gp_model(datadir):
    # ToDo
    # What if we are in the multi-shell case ?
    # Assume single shell case for now
    num_gradients = 1
    # a = np.zeros(num_gradients)  # acquisition parameters
    # h = nib.load()  # Susceptibility induced off‐resonance field (Hz)
    # ToDo
    # Provide proper values/estimates for these
    a = 1
    h = 1  # should be a NIfTI image

    # ToDo
    # Build the kernel properly following the paper.
    # Also, needs to be a sklearn.gaussian_process.kernels.Kernel instance:
    # https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Kernel.html
    # kernel = np.ones(20)
    kernel = DotProduct() + WhiteKernel()

    dwi = DWI.from_filename(datadir / "dwi.h5")

    _dwi_data = dwi.dataobj
    # Use a subset of the data for now to see that something is written to the
    # output
    # bvecs = dwi.gradients[:3, :].T
    bvecs = dwi.gradients[:3, 10:13].T  # b0 values have already been masked
    # bvals = dwi.gradients[3:, 10:13].T  # Only for inspection purposes: [[1005.], [1000.], [ 995.]]
    dwi_data = _dwi_data[60:63, 60:64, 40:45, 10:13]

    num_iterations = 5
    gp = model.GaussianProcessModel(
        dwi=dwi, a=a, h=h, kernel=kernel, num_iterations=num_iterations
    )
    indices = list(range(bvecs.shape[0]))
    # ToDo
    # This should be done within the GP model class
    # Apply lovo strategy properly
    # Vectorize and parallelize
    result = np.zeros_like(dwi_data)
    for idx in indices:
        lovo_idx = np.ones(len(indices), dtype=bool)
        lovo_idx[idx] = False
        X = bvecs[lovo_idx]
        for i in range(dwi_data.shape[0]):
            for j in range(dwi_data.shape[1]):
                for k in range(dwi_data.shape[2]):
                    # ToDo
                    # Use a mask to avoid traversing background data
                    y = dwi_data[i, j, k, lovo_idx]
                    gp.fit(X, y)
                    prediction, _ = gp.predict(
                        bvecs[idx, :][np.newaxis]
                    )  # Can take multiple values X[:2, :]
                    result[i, j, k, idx] = prediction.item()

    assert result.shape == dwi_data.shape


def test_two_initialisations(datadir):
    """Check that the two different initialisations result in the same models"""

    # Load test data
    dmri_dataset = DWI.from_filename(datadir / "dwi.h5")

    # Split data into test and train set
    data_train, data_test = lovo_split(dmri_dataset, 10)

    # Direct initialisation
    model1 = model.AverageDWModel(
        S0=dmri_dataset.bzero,
        th_low=100,
        th_high=1000,
        bias=False,
        stat="mean",
    )
    model1.fit(data_train[0], gtab=data_train[1])
    predicted1 = model1.predict(data_test[1])

    # Initialisation via ModelFactory
    model2 = model.ModelFactory.init(
        gtab=data_train[1],
        model="avg",
        S0=dmri_dataset.bzero,
        th_low=100,
        th_high=1000,
        bias=False,
        stat="mean",
    )
    model2.fit(data_train[0], gtab=data_train[1])
    predicted2 = model2.predict(data_test[1])

    assert np.all(predicted1 == predicted2)
