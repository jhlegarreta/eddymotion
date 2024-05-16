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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY kIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
import numpy as np
from dipy.core.gradients import get_bval_indices
from sklearn.cluster import KMeans

B0_THRESHOLD = 50  # from dmriprep
SHELL_DIFF_THRES = 20  # 150 in dmriprep


def is_positive_definite(matrix):
    """Check whether the given matrix is positive definite. Any positive
    definite matrix can be decomposed as the product of a lower triangular
    matrix and its conjugate transpose by performing the Cholesky decomposition.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to check.

    Returns
    -------
    True is the matrix is positive definite; False otherwise
    """

    try:
        # Attempt Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        # Matrix is not positive definite
        return False


def variance(data, indx):
    m = np.mean(data[indx])
    v = np.sum(np.square(data[indx] - m))
    return v / (len(indx) - 1)


def compute_hyperparams_spherical_matrix(data):
    """Used for single-shell data. SphericalKMatrix::GetHyperParGuess"""

    num_groups = len(grps)

    # Calculate group-wise variances (across directions) averaged over voxels
    vars = np.zeros(num_groups)
    for i in range(num_groups):
        for j in range(len(data)):
            vars[i] += variance(data[j], grps[i])
        vars[i] /= len(data)

    # Make (semi-educated) guesses for hyperparameters
    hpar = [0.0] * n_par(num_groups)
    sn_wgt = 0.25
    cntr = 0

    # ToDo
    # Vectorize this
    for cntr in range(100):
        for j in range(num_groups):
            for i in range(j, num_groups):
                pi = ij_to_parameter_index(i, j, num_groups)
                if i == j:
                    hpar[pi] = np.log(vars[i] / 2.7)
                    hpar[pi + 1] = 1.5
                    hpar[pi + 2] = np.log(sn_wgt * vars[i])
                else:
                    hpar[pi] = np.log(0.8 * min(vars[i], vars[j]) / 2.7)
                    hpar[pi + 1] = 1.5

        if not valid_hpars(hpar):
            sn_wgt *= 1.1
        else:
            break

    if cntr == 99:
        raise ValueError("Unable to find valid hyperparameters")

    return hpar


def compute_hyperparams_newspherical_matrix(data):
    """Used for multi-shell data. NewSphericalKMatrix::GetHyperParGuess"""

    num_groups = len(grps)
    vars = np.zeros(num_groups)
    var = 0.0

    # ToDo
    # Vectorize this
    for i in range(num_groups):
        for j in range(len(data)):
            vars[i] += variance(data[j], grps[i])
        vars[i] /= len(data)
        var += vars[i]

    var /= num_groups

    # Make (semi-educated) guesses for hyperparameters based on findings in GP
    # paper
    hpar = [0.0] * n_par(num_groups)
    hpar[0] = 0.9 * np.log(var / 3.0)
    hpar[1] = 0.45

    if num_groups == 1:
        hpar[2] = np.log(var / 3.0)
    else:
        hpar[2] = 0.0
        delta = 0.2 / (num_groups - 1)
        # ToDo
        # Vectorize this
        for i in range(num_groups):
            hpar[3 + i] = (1.1 - i * delta) * np.log(var / 3.0)

    if not valid_hpars(hpar):
        raise ValueError("Unable to find valid hyperparameters")

    return hpar


def calculate_angle_matrix(bvecs):
    """Matrix of angles between gradients.
    FSL crew takes as input dpars, which are diffusion parameters"""

    # ToDo
    # Does this deal with multi-shell? What does the matrix look like cols vs rows?
    # Should the b0 be included?
    # What if whe have different bvals ?
    # Note that the double loops assume bvecs is a 2D array

    # Check if bvecs is a NumPy array, list, or pandas DataFrame
    if isinstance(bvecs, list):
        bvecs = np.array(bvecs)
    elif isinstance(bvecs, pd.DataFrame):
        bvecs = bvecs.to_numpy()
    elif not isinstance(bvecs, np.ndarray):
        raise ValueError("Input bvecs must be a NumPy array, list, or pandas DataFrame")

    # Check the dimensions of bvecs
    if bvecs.shape[0] != 3:
        if bvecs.shape[1] == 3:
            bvecs = bvecs.T
        else:
            raise ValueError("bvecs must be of shape (3, n) or (n, 3)")

    n = bvecs.shape[1]
    angle_mat = np.zeros((n, n))
    
    for j in range(n):
        for i in range(j, n):
            dot_product = np.dot(bvecs[:, i], bvecs[:, j])
            dot_product = min(1.0, max(-1.0, dot_product))  # Clamp the value to be within [-1, 1]
            angle_mat[i, j] = np.arccos(np.abs(dot_product))
            angle_mat[j, i] = angle_mat[i, j]
            
    return angle_mat


def compute_exponential_function(th, a):
    """Compute the exponential function according to eq. 9 in [Andersson15]_.

    .. math::

    C(\theta) =  \exp(- \frac{\theta}{a})

    Parameters
    ----------
    th : np.ndarray
        .
    a : float > 0
        Positive scale parameter that here determines the "distance" at which θ
        the covariance goes to zero.

    Returns
    -------
        The spherical function.

    References
    ----------
    .. [Andersson15] J. L. R. Andersson. et al., Non-parametric representation
    and prediction of single- and multi-shell diffusion-weighted MRI data using
    Gaussian processes, NeuroImage 122 (2015) 166–176
    """

    assert a > 0
    return np.exp(-th / a)


def compute_spherical_function(theta, a):
    """Compute the spherical function according to eq. 10 in [Andersson15]_.

    .. math::

    C(\theta) = \begin{cases}
      1 - \frac{3 \theta}{2 a} + \frac{\theta^3}{2 a^3} & \textnormal{if} \; \theta \leq a \\
      0 & \textnormal{if} \; \theta > a
    \end{cases}

    Parameters
    ----------
    theta : np.ndarray
        .
    a : float > 0
        Positive scale parameter that here determines the "distance" at which θ
        the covariance goes to zero.

    Returns
    -------
        The spherical function.

    References
    ----------
    .. [Andersson15] J. L. R. Andersson. et al., Non-parametric representation
    and prediction of single- and multi-shell diffusion-weighted MRI data using
    Gaussian processes, NeuroImage 122 (2015) 166–176
    """

    assert a > 0
    return 1.0 - 1.5 * theta / a + 0.5 * (theta ** 3) / (a ** 3)


def compute_squared_exponential_function(grpi, grpb, l):
    """Compute the squared exponential smooth function describing how the
    covariance changes along the b direction.

    It uses the log of the b-values as the measure of distance along the
    b-direction according to eq. 15 in [Andersson15]_.

    .. math::

    C_{b}(b, b'; \ell) = \exp\left( - \frac{(\log b - \log b')^2}{2 \ell^2} \right)

    Parameters
    ----------
    grpi : np.ndarray
        Group of indices.
    grpb : np.ndarray
        Groups of b-values.
    l : float

    Returns
    -------
        The squared exponential function.

    References
    ----------
    .. [Andersson15] J. L. R. Andersson. et al., Non-parametric representation
    and prediction of single- and multi-shell diffusion-weighted MRI data using
    Gaussian processes, NeuroImage 122 (2015) 166–176
    """

    # Compute log probability of b-values
    log_grpb = np.log(grpb)
    bv_diff = log_grpb[grpi[:, None]] - log_grpb[grpi]
    return np.exp(-(bv_diff ** 2) / (2 * l ** 2))

# ToDo
# Long-term: generalize this so that the loop block can be put into a function
# and be called from the single-shell and multi-shell
def compute_single_shell_covariance_matrix(angle_mat, grpi, ngrp, thpar):
    """Compute single-shell covariance.
    SphericalKMatrix::calculate_K_matrix"""

    k = np.zeros_like(angle_mat)

    # Compute angular covariance
    # ToDo
    # Vectorize this
    for j in range(k.shape[1]):
        for i in range(j, k.shape[0]):
            pindx = ij_to_parameter_index(grpi[i], grpi[j], ngrp)
            sm = thpar[pindx]
            a = thpar[pindx + 1]
            theta = angle_mat[i, j]

            # eq. 10 in [Andersson15]_
            if a > theta:
                k[i+1, j+1] = sm * compute_spherical_function(theta, a)
            else:
                k[i+1, j+1] = 0.0
            if i == j:
                k[i+1, j+1] += thpar[pindx + 2]
    return k

def compute_multi_shell_covariance_matrix1(k, thpar, grpb, grpi):
    """Compute multi-shell covariance.
    Indicies2KMatrix::common_construction
    hpar are hyperparameters; thpar are "transformed" (for example
    exponentiated) hyperparameters
    """

    sm = thpar[0]
    a = thpar[1]
    l = thpar[2]

    # Make angle matrix (eq. 11 in [Andersson15]_)
    # ToDo
    # Vectorize this
    # _k.resize(nval, nval); missing
    for j in range(nval):
        for i in range(j, nval):
            if i == j:
                k[i, j] = 0.0  # set diagonal elements to 0
            else:
                k[i, j] = np.arccos(min(1.0, abs(np.dot(bvecs[i], bvecs[j]))))
                k[j, i] = k[i, j]  # make it symmetric

    # Compute angular covariance
    # ToDo
    # Vectorize this
    for j in range(k.shape[1]):
        for i in range(j, k.shape[0]):
            theta = k[i+1, j+1]
            # eq. 10 in [Andersson15]_
            if a > theta:
                k[i+1, j+1] = sm * (1.0 - 1.5 * theta / a + 0.5 * (theta ** 3) / (a ** 3))
            else:
                k[i+1, j+1] = 0.0
            if i != j:
                k[j+1, i+1] = k[i+1, j+1]  # make it symmetric

    # Compute b-value covariance
    # ToDo
    # Vectorize this
    if len(grpb) > 1:
        log_grpb = np.log(grpb)
        # Takes upper triangular elements
        for j in range(k.shape[1]):
            for i in range(j + 1, k.shape[0]):
                if k[i+1, j+1] != 0.0:
                    bvdiff = log_grpb[grpi[i]] - log_grpb[grpi[j]]
                    if bvdiff:
                        # eq. 15 in [Andersson15]_
                        k[i+1, j+1] *= np.exp(-(bvdiff ** 2) / (2 * l ** 2))
                        k[j+1, i] = k[i+1, j+1]  # make it symmetric


def compute_multi_shell_covariance_matrix2(k, angle_mat, thpar, grpb, grpi):
    """Compute multi-shell covariance.
    NewSphericalKMatrix::calculate_K_matrix
    """

    sm = thpar[0]
    a = thpar[1]
    l = thpar[2]

    # Compute angular covariance
    # ToDo
    # Vectorize this
    for j in range(K.shape[1]):
        for i in range(j, K.shape[0]):
            theta = angle_mat[i+1, j+1]
            if a > theta:
                K[i+1, j+1] = sm * (1.0 - 1.5 * theta / a + 0.5 * (theta ** 3) / (a ** 3))
            else:
                K[i+1, j+1] = 0.0

    # Compute b-value covariance
    # ToDo
    # Vectorize this
    if ngrp > 1:
        log_grpb = np.log(grpb())
        for j in range(K.shape[1]):
            for i in range(j + 1, K.shape[0]):
                bvdiff = log_grpb[grpi[i]] - log_grpb[grpi[j]]
                if bvdiff:
                    K[i+1, j+1] *= np.exp(-(bvdiff ** 2) / (2 * l ** 2))


# ToDo
# Naming: DWI vs dMRI
def extract_dmri_shell(dwi, bvals, bvecs, bvals_to_extract, tol=SHELL_DIFF_THRES):
    """Extract the DWI volumes that are on the given b-value shells. Multiple
    shells can be extracted at once by specifying multiple b-values. The
    extracted volumes will be in the same order as in the original file.

    Parameters
    ----------
    dwi : nib.Nifti1Image
        Original dMRI multi-shell volume.
    bvals : ndarray
         b-values in FSL format.
    bvecs : ndarray
        b-vectors in FSL format.
    bvals_to_extract : list of int
        List of b-values to extract.
    tol : int, optional
        Tolerance between the b-values to extract and the actual b-values.

    Returns
    -------
    indices : ndarray
        Indices of the volumes corresponding to the given ``bvals``.
    shell_data : ndarray
        Volumes corresponding to the given ``bvals``.
    output_bvals : ndarray
        Selected b-values (as extracted from ``bvals``).
    output_bvecs : ndarray
        Selected b-vectors.
    """

    indices = [
        get_bval_indices(bvals, shell, tol=tol) for shell in bvals_to_extract
    ]
    indices = np.unique(np.sort(np.hstack(indices)))

    if len(indices) == 0:
        raise ValueError(
            f"No dMRI volumes found corresponding to the given b-values: {bvals_to_extract}"
        )

    shell_data = dwi.get_fdata()[..., indices]
    output_bvals = bvals[indices].astype(int)
    output_bvecs = bvecs[indices, :]

    return indices, shell_data, output_bvals, output_bvecs


# ToDo
# Long term: use DiffusionGradientTable from dmriprep, normalizing gradients,
# etc. ?
def find_shelling_scheme(bvals, tol=SHELL_DIFF_THRES):
    """Find the shelling scheme on the given b-values: extract the b-value
    shells as the b-values centroids using k-means clustering.

    Parameters
    ----------
    bvals : ndarray
         b-values in FSL format.
    tol : int, optional
        Tolerance between the b-values and the centroids in the average squared
        distance sense.

    Returns
    -------
    shells : ndarray
        b-value shells.
    bval_centroids : ndarray
        Shell value corresponding to each value in ``bvals``.
    """

    # Use kmeans to find the shelling scheme
    for k in range(1, len(np.unique(bvals)) + 1):
        kmeans_res = KMeans(n_clusters=k).fit(bvals.reshape(-1, 1))
        # ToDo
        # The tolerance is not a very intuitive value, as it has to do with the
        # sum of squared distances across all samples to the centroids
        # (_inertia)
        # Alternatives:
        # - We could accept the number of clusters as a parameter and do
        # kmeans_res = KMeans(n_clusters=n_clusters)
        # Setting that to 3 in the last testing case, where tol = 60 is not
        # intuitive would give the expected 6, 1000, 2000 clusters.
        # Passes all tests. But maybe not tested corner cases
        # We could have both k and tol as optional parameters, set to None by
        # default to force the user set one
        # - Use get_bval_indices to get the cluster centroids and then
        # substitute the values in bvals with the corresponding values
        # indices = [get_bval_indices(bvals, shell, tol=tol) for shell in bvals_to_extract]
        # result = np.zeros_like(bvals)
        # for i, idx in enumerate(indices):
        #   result[idx] = bvals_to_extract[i]

        if kmeans_res.inertia_ / len(bvals) < tol:
            break
    else:
        raise ValueError(
            f"bvals parsing failed: no shells found more than {tol} apart"
        )

    # Convert the kclust labels to an array
    shells = kmeans_res.cluster_centers_
    bval_centroids = np.zeros(bvals.shape)
    for i in range(shells.size):
        bval_centroids[kmeans_res.labels_ == i] = shells[i][0]

    return np.sort(np.squeeze(shells, axis=-1)), bval_centroids
