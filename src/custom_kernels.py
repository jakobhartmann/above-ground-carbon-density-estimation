# source: https://gpy.readthedocs.io/en/deploy/tuto_creating_new_kernels.html

import numpy as np
from GPy.kern import Kern
from GPy.kern.src.rbf import RBF
from GPy.core.parameterization import Param

class WaterRBFKernel(Kern):
    def __init__(self, input_dim, variance_land, lengthscale_land, bitmask_land_land, bitmask_water_water, active_dims = None):
        super(WaterRBFKernel, self).__init__(input_dim, active_dims, 'water_rbf_kernel')
        assert input_dim == 1 # IMPORTANT!!!
        self.variance_land = Param('variance_land', variance_land)
        self.lengthscale_land = Param('lengthscale_land', lengthscale_land)
        self.rbf_kernel_land = RBF(input_dim = input_dim, variance = self.variance_land, lengthscale = self.lengthscale_land)
        self.bitmask_land_land = bitmask_land_land
        self.bitmask_water_water = bitmask_water_water
        self.link_parameters(self.variance_land, self.lengthscale_land)

    def parameters_changed(self):
        pass

    def K(self, X, X2):
        if X2 is None:
            X2 = X
        # Calculate covariance matrix for all points using land RBF kernel
        K_land_land = self.rbf_kernel_land.K(X, X2)
        # Mask out all entries that do not correspond to two land points
        K_land_land = K_land_land * self.bitmask_land_land
        # Get max value of land covariance matrix
        max_value = np.max(K_land_land)
        # Initialize water covariance matrix
        K_water_water = np.copy(self.bitmask_water_water)
        # Set variance of each water point to zero (no uncertainty)
        np.fill_diagonal(K_water_water, 0)
        # Set covariance of two water points to the max value of land kernel (high similarity)
        K_water_water = K_water_water * max_value
        # Return sum of land and water covariance matrix (covariance between land and water is assumed to be 0)
        return K_land_land + K_water_water
    
    def Kdiag(self, X):
        # Return diagonal of masked-out land covariance matrix
        # Variance of each water point is zero and land <-> water points do not exist on diagonal
        return self.rbf_kernel_land.Kdiag(X) * self.bitmask_land_land.diagonal()