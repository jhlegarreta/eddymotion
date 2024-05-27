import numpy as np
import pytest

from eddymotion.model.utils import SphericalCovarianceKernel

def test_kernel_call():
    # Create a SphericalCovarianceKernel instance
    kernel = SphericalCovarianceKernel(lambda_=2.0, a=1.0, sigma_sq=0.5)

    # Create trivial data (pairwise angles)
    theta = np.array([[0.0, 0.5, 1.0],
                      [0.5, 0.0, 0.5],
                      [1.0, 0.5, 0.0]])

    # Expected kernel matrix
    expected_K = np.array([[2.5, 2.0 * (1 - 3 * (0.5 / 1.0) ** 2 + 2 * (0.5 / 1.0) ** 3), 0.0],
                           [2.0 * (1 - 3 * (0.5 / 1.0) ** 2 + 2 * (0.5 / 1.0) ** 3), 2.5, 2.0 * (1 - 3 * (0.5 / 1.0) ** 2 + 2 * (0.5 / 1.0) ** 3)],
                           [0.0, 2.0 * (1 - 3 * (0.5 / 1.0) ** 2 + 2 * (0.5 / 1.0) ** 3), 2.5]])

    # Compute the kernel matrix using the kernel instance
    K = kernel(theta)

    # Assert the kernel matrix is as expected
    np.testing.assert_array_almost_equal(K, expected_K, decimal=6)

def test_kernel_diag():
    # Create a SphericalCovarianceKernel instance
    kernel = SphericalCovarianceKernel(lambda_=2.0, a=1.0, sigma_sq=0.5)

    # Create trivial data
    X = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

    # Expected diagonal
    expected_diag = np.array([2.5, 2.5, 2.5])

    # Compute the diagonal using the kernel instance
    diag = kernel.diag(X)

    # Assert the diagonal is as expected
    np.testing.assert_array_almost_equal(diag, expected_diag, decimal=6)

if __name__ == "__main__":
    pytest.main()