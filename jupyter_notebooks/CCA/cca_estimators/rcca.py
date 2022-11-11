import warnings

import numpy as np
from numpy.linalg import LinAlgError
import scipy
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.metrics.pairwise import pairwise_kernels, cosine_similarity
from sklearn.preprocessing import StandardScaler

class TwoViewRCCA(BaseEstimator, MultiOutputMixin):
    """Two-view Regularized CCA Class

    """
    def __init__(
            self,
            n_components: int = 1,
            c_x=1.0,
            c_y=1.0,
            solving_method='svd',
            random_state=None,
            standardization: bool = True
    ):
        self.n_components = n_components
        self.c_x = c_x
        self.c_y = c_y
        self.solving_method = solving_method
        self.random_state = random_state
        self.standardization = standardization

    def fit(self, X, y, **kwargs):
        # check if X and y have the same sizes
        [X, y] = self._check_x_y(X, y)

        # extract sizes
        n_sample = X.shape[0]
        n_x_dim = X.shape[1]
        n_y_dim = y.shape[1]
        if np.min([n_x_dim, n_y_dim]) < self.n_components:
            raise ValueError('Please make sure the n_components is smaller than the min(n_x_dim, n_y_dim).')

        # standardization of data
        if self.standardization:
            self.scaler_x_ = StandardScaler().fit(X)
            self.scaler_y_ = StandardScaler().fit(y)
            X = self.scaler_x_.transform(X)
            y = self.scaler_y_.transform(y)

        # compute joint covariance matrix
        self.C_ = np.cov(X.T, y.T, ddof=0)
        self.Cxx_ = self.C_[:n_x_dim, :n_x_dim]
        self.Cxy_ = self.C_[:n_x_dim, n_x_dim:]
        self.Cyy_ = self.C_[n_x_dim:, n_x_dim:]

        # regularization
        self.Cxx_r_ = self.Cxx_ + self.c_x * np.eye(n_x_dim)
        self.Cyy_r_ = self.Cyy_ + self.c_y * np.eye(n_y_dim)

        if self.solving_method == 'svd':
            # try to solve CCA through svd method
            try:
                self.Cxx_r_sqrt_inv_ = scipy.linalg.inv(scipy.linalg.sqrtm(self.Cxx_r_))
                self.Cyy_r_sqrt_inv_ = scipy.linalg.inv(scipy.linalg.sqrtm(self.Cyy_r_))
                self.mat_to_svd_ = np.linalg.multi_dot(
                    [self.Cxx_r_sqrt_inv_, self.Cxy_, self.Cyy_r_sqrt_inv_]
                )

                # SVD decomposition
                [self.U_, self.cc_arr_, self.Vh_] = scipy.linalg.svd(self.mat_to_svd_, full_matrices=False)

                # compute weights vectors
                self.w_x_ = np.matmul(self.Cxx_r_sqrt_inv_, self.U_)
                self.w_y_ = np.matmul(self.Cyy_r_sqrt_inv_, self.Vh_.T)

                # slicing arrays
                self.cc_arr_ = self.cc_arr_[:self.n_components]
                self.w_x_ = self.w_x_[:, :self.n_components]
                self.w_y_ = self.w_y_[:, :self.n_components]
            except LinAlgError:
                warnings.warn('LinAlgError for bad parameters, 0.0 eigenvalue and eigenvectors will be provided.')
                self.cc_arr_ = np.zeros((self.n_components, ))
                self.w_x_ = np.zeros((n_x_dim, self.n_components))
                self.w_y_ = np.zeros((n_y_dim, self.n_components))
        elif self.solving_method == 'gen_eig':
            # prepare for solving the generalized eigenvalue problem
            self.A_ = np.zeros((n_x_dim + n_y_dim, n_x_dim + n_y_dim))
            self.A_[:n_x_dim, n_x_dim:] = self.Cxy_
            self.A_[n_x_dim:, :n_x_dim] = self.Cxy_.T
            self.B_ = np.zeros_like(self.A_)
            self.B_[:n_x_dim, :n_x_dim] = self.Cxx_r_
            self.B_[n_x_dim:, n_x_dim:] = self.Cyy_r_

            # try to solve CCA through generalized eigenvalue problem
            try:
                [eig_val, eig_vec] = scipy.linalg.eigh(self.A_, self.B_)
                # sort by eigenvalues, descending order
                sort_ind_des = eig_val.argsort()[::-1]
                eig_val_sorted = eig_val[sort_ind_des]
                eig_vec_sorted = eig_vec[:, sort_ind_des]

                # slicing arrays, up to the components
                self.eig_val_ = eig_val_sorted[:self.n_components]
                self.eig_vec_ = eig_vec_sorted[:, :self.n_components]
                self.w_x_ = self.eig_vec_[:n_x_dim, :]
                self.w_y_ = self.eig_vec_[n_x_dim:, :]
            except LinAlgError:
                warnings.warn('LinAlgError for bad parameters, 0.0 eigenvalue and eigenvectors will be provided.')
                self.eig_val_ = np.zeros((self.n_components, ))
                self.eig_vec_ = np.zeros((n_x_dim + n_y_dim, self.n_components))
                self.w_x_ = np.zeros((n_x_dim, self.n_components))
                self.w_y_ = np.zeros((n_y_dim, self.n_components))
        else:
            raise TypeError('Please provide correct method between svd, gen_eig')

        # transform matrices
        self.z_x_ = X.dot(self.w_x_)
        self.z_y_ = y.dot(self.w_y_)
        return self

    def transform(self, X, y):
        # check if fit is called
        check_is_fitted(self)

        if y is None:
            X = check_array(X)
            z_x = X.dot(self.w_x_)
            return z_x
        else:
            self._check_x_y(X, y)
            z_x = X.dot(self.w_x_)
            z_y = y.dot(self.w_y_)
            return [z_x, z_y]

    def score(self, X, y):
        # check if fit is called
        check_is_fitted(self)

        z_x_score = X.dot(self.w_x_)
        z_y_score = y.dot(self.w_y_)

        if self.n_components == 1:
            return cosine_similarity(
                z_x_score.reshape(1, -1),
                z_y_score.reshape(1, -1)
            )[0][0]
        else:
            return np.diag(
                cosine_similarity(z_x_score.T, z_y_score.T)
            )

    def _check_x_y(self, X, y):
        if (type(y) is list) or (type(y) is tuple):
            y = np.array(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError('Please make sure X, y have the same sample number')
        return [check_array(X), check_array(y)]

    def predict(self, X):
        """This method is not in use.

        Parameters
        ----------
        X

        Returns
        -------

        """
        # check if fit is called
        check_is_fitted(self)
        return X