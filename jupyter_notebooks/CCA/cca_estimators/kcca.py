import warnings

import numpy as np
from numpy.linalg import LinAlgError
import scipy
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
from sklearn.metrics.pairwise import pairwise_kernels, cosine_similarity
from sklearn.preprocessing import StandardScaler


class TwoViewKCCA(BaseEstimator, MultiOutputMixin):
    """Two-view Kernel CCA class.

    """
    def __init__(
            self,
            n_components: int = 1,
            kernel_name: str = 'linear',
            c_x=1.0,
            c_y=1.0,
            gamma_x=0.5,
            gamma_y=0.5,
            degree_x=1,
            degree_y=1,
            coef0_x=0,
            coef0_y=0,
            random_state=None,
            standardization: bool = True
    ):
        self.n_components = n_components
        self.kernel_name = kernel_name
        self.c_x = c_x
        self.c_y = c_y
        self.gamma_x = gamma_x
        self.gamma_y = gamma_y
        self.degree_x = degree_x
        self.degree_y = degree_y
        self.coef0_x = coef0_x
        self.coef0_y = coef0_y
        self.random_state = random_state
        self.standardization = standardization

    def fit(self, X, y, **kwargs):
        # check if X and y have the same sizes
        [X, y] = self._check_x_y(X, y)

        # extract sizes
        n_sample = X.shape[0]
        if n_sample < self.n_components:
            raise ValueError('Please make sure the n_components is smaller than the sample number.')

        # standardization of data
        if self.standardization:
            self.scaler_x_ = StandardScaler().fit(X)
            self.scaler_y_ = StandardScaler().fit(y)
            X = self.scaler_x_.transform(X)
            y = self.scaler_y_.transform(y)

        # prepare for score and transform functions
        self.fit_X = X.copy()
        self.fit_y = y.copy()

        # compute kernels
        self.K_x_ = self.kernel_computing(X, X, 'x')
        self.K_y_ = self.kernel_computing(y, y, 'y')
        Kxy = self.K_x_.dot(self.K_y_)

        self.A_ = np.zeros((2 * n_sample, 2 * n_sample))
        self.A_[:n_sample, n_sample:] = Kxy
        self.A_[n_sample:, :n_sample] = Kxy.T

        self.B_ = np.zeros_like(self.A_)
        self.B_[:n_sample, :n_sample] = np.linalg.matrix_power(
            self.K_x_ + self.c_x * np.eye(n_sample), 2
        )
        self.B_[n_sample:, n_sample:] = np.linalg.matrix_power(
            self.K_y_ + self.c_y * np.eye(n_sample), 2
        )

        # try if B is positive definite
        try:
            # solve for generalized eigenvalue problem
            [eig_val, eig_vec] = scipy.linalg.eigh(self.A_, self.B_)

            # sort by eigenvalues, descending order
            sort_ind_des = eig_val.argsort()[::-1]
            eig_val_sorted = eig_val[sort_ind_des]
            eig_vec_sorted = eig_vec[:, sort_ind_des]

            # slicing arrays, up to the components
            self.eig_val_ = eig_val_sorted[:self.n_components]
            self.eig_vec_ = eig_vec_sorted[:, :self.n_components]
            self.alpha_ = self.eig_vec_[:n_sample, :]
            self.beta_ = self.eig_vec_[n_sample:, :]
        except LinAlgError:
            warnings.warn('LinAlgError for bad parameters, 0.0 eigenvalue and eigenvectors will be provided.')
            self.eig_val_ = np.zeros((self.n_components, ))
            self.alpha_ = np.zeros((n_sample, self.n_components))
            self.beta_ = np.zeros_like(self.alpha_)

        # transform kernels
        self.z_x_ = self.K_x_.dot(self.alpha_)
        self.z_y_ = self.K_y_.dot(self.beta_)
        return self

    def kernel_computing(self, a, b, x_y_indicator: str = 'x'):
        params = {
            'gamma': getattr(self, 'gamma_'+x_y_indicator),
            'degree': getattr(self, 'degree_'+x_y_indicator),
            'coef0': getattr(self, 'coef0_'+x_y_indicator)
        }
        return pairwise_kernels(
            a, b, metric=self.kernel_name, filter_params=True, **params
        )

    def kernel_computing_w_fit(self, X, y=None):
        # check if fit is called
        check_is_fitted(self)

        self._check_x_y(X, y)
        if y is None:
            if self.standardization:
                X = self.scaler_x_.transform(X)
            return self.kernel_computing(X, self.fit_X, 'x')
        else:
            if self.standardization:
                X = self.scaler_x_.transform(X)
                y = self.scaler_y_.transform(y)
            return [self.kernel_computing(X, self.fit_X, 'x'), self.kernel_computing(y, self.fit_y, 'y')]

    def transform(self, X, y=None):
        # check if fit is called
        check_is_fitted(self)
        self._check_x_y(X, y)

        if y is None:
            K_x = self.kernel_computing_w_fit(X, y)
            z_x = K_x.dot(self.alpha_)
            return z_x
        else:
            K_x, K_y = self.kernel_computing_w_fit(X, y)
            z_x = K_x.dot(self.alpha_)
            z_y = K_y.dot(self.beta_)
            return [z_x, z_y]

    def score(self, X, y):
        # check if fit is called
        check_is_fitted(self)

        Kx_score, Ky_score = self.kernel_computing_w_fit(X, y)
        z_x_score = Kx_score.dot(self.alpha_)
        z_y_score = Ky_score.dot(self.beta_)

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



