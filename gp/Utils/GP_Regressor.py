from sklearn.gaussian_process import GaussianProcessRegressor

######################################################################################################
# GP_Regressor Class as an implementation of the Gaussian Process Implicit Surface GPIS
# Includes the following functions
#       - fit
#       - predict
######################################################################################################
class GP_Regressor:
    def __init__(self, kernel, alpha, optimizer='fmin_l_bfgs_b', random_state=4):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.random_state = random_state
        self.gp = None

    def fit(self, X_train, y_train):
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            optimizer=self.optimizer,
            random_state=self.random_state
        )
        self.gp.fit(X_train, y_train)

    def predict(self, Xstar, return_cov=False):
        return self.gp.predict(Xstar, return_cov=return_cov)