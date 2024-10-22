from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

######################################################################################################
# GP_Regressor Class as an implementation of the Gaussian Process Implicit Surface GPIS
# Includes the following functions
#       - fit
#       - predict
######################################################################################################
class GP_Regressor:
    def __init__(self, length_scale=0.04, alpha=0.001**2, optimizer='fmin_l_bfgs_b', random_state=4):
        """
        Initializes the GP_Regressor object.

        Parameters
        ----------
            length_scale: The length_scale for the kernel specifying the covariance function.
            alpha (float): Value added to the diagonal of the kernel matrix during fitting.
            optimizer (str): The optimizer to use during fitting.
            random_state (int): Seed used by the random number generator.
        """
        self.alpha = alpha
        self.optimizer = optimizer
        self.random_state = random_state
        self.gp = None
        self.kernel = 1.0 * RBF(length_scale=length_scale) 


    def fit(self, X_train, y_train):
        """
        Fits the Gaussian Process Regressor to the training data.

        Parameters
        ----------
            X_train (ndarray): The input training data.
            y_train (ndarray): The target training data.
        """
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            optimizer=self.optimizer,
            random_state=self.random_state
        )
        self.gp.fit(X_train, y_train)


    def predict(self, Xstar, return_cov=False):
        """
        Predicts target values for the given test data.

        Parameters
        ----------
            Xstar (ndarray): The input test data.
            return_cov (bool): Whether to return the covariance of the prediction.

        Returns
        -------
            ndarray: Predicted target values.
        """
        return self.gp.predict(Xstar, return_cov=return_cov)