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


def test_gp_model():

    import nibabel as nib
    # ToDo
    # Transition to the HDF5 format eventually
    root_path = "nireports/nireports/tests/data"
    fname_dwi = root_path + "/ds000114_sub-01_ses-test_desc-trunc_dwi.nii.gz"
    fname_bval = root_path + "/ds000114_sub-01_ses-test_desc-trunc_dwi.bval"
    fname_bvec = root_path + "/ds000114_sub-01_ses-test_desc-trunc_dwi.bvec"
    dwi_img = nib.load(fname_dwi)
    bvecs = np.loadtxt(fname_bvec).T
    bvals = np.loadtxt(fname_bval)
    b0s_mask = bvals < 50
    gradients = np.hstack([bvecs[~b0s_mask], bvals[~b0s_mask, None]])
    # ToDo: what if we are in the multi-shell case ?
    # Assume single shell case for now
    num_gradients = 1
    # a = np.zeros(num_gradients)  # acquisition parameters
    # h = nib.load()  # Susceptibility induced off‐resonance field (Hz)
    # ToDo
    # Provide proper values/estimates for these
    a = 1
    h = 1  # should be a nifti image

    # ToDo
    # Build kernel properly
    kernel = np.ones(20)

    num_iterations = 5
    gp = model.GaussianProcessModel(dwi=dwi_img, a=a, h=h, kernel=kernel, num_iterations=num_iterations)

    dwi_data = dwi_img.get_fdata()[..., ~b0s_mask]

    lovo_index = 5
    train_idx = np.where(np.asarray(range(len(bvals[~b0s_mask]))) != lovo_index)
    data_train = dwi_data[..., train_idx]
    data_train = dwi_data
    gp.fit(data_train[0], data_train[1])
    predicted1 = gp.predict(data_test[1])


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
