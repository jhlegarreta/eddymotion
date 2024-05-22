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
import nibabel as nib
import numpy as np
from eddymotion.model.utils import (
    extract_dmri_shell,
    find_shelling_scheme,
    is_positive_definite,
    # update_covariance1,
    # update_covariance2,
)


def test_is_positive_definite():

    matrix = np.array([[4, 1, 2], [1, 3, 1], [2, 1, 5]])
    assert is_positive_definite(matrix)

    matrix = np.array([[4, 1, 2], [1, -3, 1], [2, 1, 5]])
    assert not is_positive_definite(matrix)


def test_update_covariance():

    _K = np.random.rand(5, 5)
    _thpar = [0.5, 1.0, 2.0]
    update_covariance1(_K, _thpar)
    print(_K)  # Updated covariance matrix


def test_extract_dmri_shell():

    # dMRI volume with 5 gradients
    bvals = np.asarray([0, 1980, 12, 990, 2000])
    bval_count = len(bvals)
    vols_size = (10, 15, 20)
    dwi = np.ones((*vols_size, bval_count))
    bvecs = np.ones((bval_count, 3))
    # Set all i-th gradient dMRI volume data and bvecs values to i
    for i in range(bval_count):
        dwi[..., i] = i
        bvecs[i, :] = i
    dwi_img = nib.Nifti1Image(dwi, affine=np.eye(4))

    bvals_to_extract = [0, 2000]
    tol = 15

    expected_indices = np.asarray([0, 2, 4])
    expected_shell_data = np.stack([i*np.ones(vols_size) for i in expected_indices], axis=-1)
    expected_shell_bvals = np.asarray([0, 12, 2000])
    expected_shell_bvecs = np.asarray([[i]*3 for i in expected_indices])

    (
        obtained_indices,
        obtained_shell_data,
        obtained_shell_bvals,
        obtained_shell_bvecs
    ) = extract_dmri_shell(
        dwi_img, bvals, bvecs, bvals_to_extract=bvals_to_extract, tol=tol)

    assert np.array_equal(obtained_indices, expected_indices)
    assert np.array_equal(obtained_shell_data, expected_shell_data)
    assert np.array_equal(obtained_shell_bvals, expected_shell_bvals)
    assert np.array_equal(obtained_shell_bvecs, expected_shell_bvecs)

    bvals = np.asarray([0, 1010, 12, 990, 2000])
    bval_count = len(bvals)
    vols_size = (10, 15, 20)
    dwi = np.ones((*vols_size, bval_count))
    bvecs = np.ones((bval_count, 3))
    # Set all i-th gradient dMRI volume data and bvecs values to i
    for i in range(bval_count):
        dwi[..., i] = i
        bvecs[i, :] = i
    dwi_img = nib.Nifti1Image(dwi, affine=np.eye(4))

    bvals_to_extract = [0, 1000]
    tol = 20

    expected_indices = np.asarray([0, 1, 2, 3])
    expected_shell_data = np.stack([i*np.ones(vols_size) for i in expected_indices], axis=-1)
    expected_shell_bvals = np.asarray([0, 1010, 12, 990])
    expected_shell_bvecs = np.asarray([[i]*3 for i in expected_indices])

    (
        obtained_indices,
        obtained_shell_data,
        obtained_shell_bvals,
        obtained_shell_bvecs
    ) = extract_dmri_shell(
        dwi_img, bvals, bvecs, bvals_to_extract=bvals_to_extract, tol=tol)

    assert np.array_equal(obtained_indices, expected_indices)
    assert np.array_equal(obtained_shell_data, expected_shell_data)
    assert np.array_equal(obtained_shell_bvals, expected_shell_bvals)
    assert np.array_equal(obtained_shell_bvecs, expected_shell_bvecs)


def test_find_shelling_scheme():

    tol = 20
    bvals = np.asarray([0, 0])
    expected_shells = np.asarray([0])
    expected_bval_centroids = np.asarray([0, 0])
    obtained_shells, obtained_bval_centroids = find_shelling_scheme(
        bvals, tol=tol)

    assert np.array_equal(obtained_shells, expected_shells)
    assert np.array_equal(obtained_bval_centroids, expected_bval_centroids)

    bvals = np.asarray([
        5, 300, 300, 300, 300, 300, 305, 1005, 995, 1000, 1000, 1005, 1000,
        1000, 1005, 995, 1000, 1005, 5, 995, 1000, 1000, 995, 1005, 995, 1000,
        995, 995, 2005, 2000, 2005, 2005, 1995, 2000, 2005, 2000, 1995, 2005, 5,
        1995, 2005, 1995, 1995, 2005, 2005, 1995, 2000, 2000, 2000, 1995, 2000, 2000,
        2005, 2005, 1995, 2005, 2005, 1990, 1995, 1995, 1995, 2005, 2000, 1990, 2010, 5
    ])
    expected_shells = np.asarray([5., 300.83333333, 999.5, 2000.])
    expected_bval_centroids = ([
        5., 300.83333333, 300.83333333, 300.83333333, 300.83333333, 300.83333333, 300.83333333, 999.5, 999.5, 999.5, 999.5, 999.5, 999.5,
        999.5, 999.5, 999.5, 999.5, 999.5, 5., 999.5, 999.5, 999.5, 999.5, 999.5, 999.5, 999.5,
        999.5, 999.5, 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 5.,
        2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000.,
        2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 2000., 5.
    ])
    obtained_shells, obtained_bval_centroids = find_shelling_scheme(
        bvals, tol=tol)

    # ToDo
    # Giving a tolerance of 15 this fails because it finds 5 clusters
    assert np.allclose(obtained_shells, expected_shells)
    assert np.allclose(obtained_bval_centroids, expected_bval_centroids)

    bvals = np.asarray([0, 1980, 12, 990, 2000])
    expected_shells = np.asarray([6, 990, 1980, 2000])
    expected_bval_centroids = np.asarray([6, 1980, 6, 990, 2000])
    obtained_shells, obtained_bval_centroids = find_shelling_scheme(
        bvals, tol=tol)

    assert np.allclose(obtained_shells, expected_shells)
    assert np.allclose(obtained_bval_centroids, expected_bval_centroids)

    bvals = np.asarray([0, 1010, 12, 990, 2000])
    tol = 60
    expected_shells = np.asarray([6, 1000, 2000])
    expected_bval_centroids = np.asarray([6, 1000, 6, 1000, 2000])
    obtained_shells, obtained_bval_centroids = find_shelling_scheme(bvals, tol)

    assert np.allclose(obtained_shells, expected_shells)
    assert np.allclose(obtained_bval_centroids, expected_bval_centroids)
