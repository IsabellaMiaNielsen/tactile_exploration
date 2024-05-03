import numpy as np
from scipy.optimize import brentq
from utils.visualizer import Visualizer as Visu

######################################################################################################
# DataAnalyzer Class for 
#       - getting the uncertainties at points on the surface (zero-crossings of the implicit function)
#       - finding the point on the surface with the highest uncertainty
# Includes the following functions
#       - get_uncertainties_at_zero_crossings
#       - find_max_uncertainty_coordinates
#       - analyze_uncertainty
#       - update_with_new_point
######################################################################################################
class DataAnalyzer:
    def __init__(self, X_train, y_train, Xstar, gp_regressor, mu_s, cov_s):
        """
        Initialize DataAnalyzer class

        Parameters
        ----------
            X_train (ndarray): Training data consists of points outside, inside and on the surface
            y_train (ndarray): Refers to the values of the implicit function.
                Assigns y(x) = +1 to points inside the surface
                Assigns y(x) = -1 to points outside the surface
                Assigns y(x) = 0 to points on the surface
            Xstar (ndarray): Refers to the grid of points between the min and max evaluation limits
            gp_regressor (class GP_Regressor): Is an instance of the GP_Regressor class that represents the GP regressor fitted to the training data. 
            mu_s (ndarray): Array containing the predicted mean by the GP regressor for each point in Xstar
            cov_s (ndarray): Covariance matrix
        """
        self.X_train = X_train
        self.y_train = y_train
        self.Xstar = Xstar
        self.gp_regressor = gp_regressor
        self.mu_s = mu_s
        self.unc = np.diag(cov_s) # Uncertainty
        self.unc_at_zero_crossings = None # Uncertainties at the zero-crossing points
        self.zero_crossings_x = None # X coordinates of the zero-crossing points
        self.zero_crossings_y = None # Y coordinates of the zero-crossing points
        self.zero_crossings_z = None # Z coordinates of the zero-crossing points
        self.max_unc_x = None
        self.max_unc_y = None
        self.max_unc_z = None
        self.max_unc = None


    def get_uncertainties_at_zero_crossings(self):
        """
        First, it finds the indices where a zero-crossing occurs (where the sign of mu_s changes).
        Then, it finds the uncertainties at the zero-crossing points based on the indices.
        """
        sign_changes = np.where(np.diff(np.sign(self.mu_s)))[0]
        zero_crossing_indices = []

        # Find the index closest to zero for each sign change
        for idx in sign_changes:
            if self.mu_s[idx] * self.mu_s[idx + 1] < 0: # Ensure sign change occurs within the interval
                zero_crossing_index = brentq(lambda x: self.mu_s[int(np.floor(x))] if x < len(self.mu_s) else self.mu_s[-1], idx, idx + 1)
                zero_crossing_indices.append(int(np.floor(zero_crossing_index)))

        zero_crossing_indices = np.array(zero_crossing_indices)

        self.zero_crossings_x = self.Xstar[zero_crossing_indices, 0]
        self.zero_crossings_y = self.Xstar[zero_crossing_indices, 1]
        self.zero_crossings_z = self.Xstar[zero_crossing_indices, 2]

        # Get uncertainties based on the indices of zero-crossing points
        self.unc_at_zero_crossings = self.unc[zero_crossing_indices]


    def find_max_uncertainty_coordinates(self, above_ground = False, z_0 = 0.0):
        """
        Finds the coordinates (x,y,z) corresponding to the maximum uncertainty estimated by the GP regressor.
        
        Parameters
        ----------
            above_ground (bool): If True, finds the maximum uncertainty where the z value is greater than z_0.
            z_0 (float): Maximum uncertainty points need to be above z_0
        """
        if above_ground:
            # Filter out zero-crossing points with z value <= z_0
            filtered_indices = np.where(self.zero_crossings_z > z_0)[0]
            filtered_uncertainties = self.unc_at_zero_crossings[filtered_indices]

            # Find the index of the maximum uncertainty
            max_index = filtered_indices[np.argmax(filtered_uncertainties)]
        else:
            # Find the index of the maximum uncertainty among all zero-crossing points
            max_index = np.argmax(self.unc_at_zero_crossings)
            
        # Retrieve the x, y, and z coordinates corresponding to the maximum uncertainty
        self.max_x = self.zero_crossings_x[max_index]
        self.max_y = self.zero_crossings_y[max_index]
        self.max_z = self.zero_crossings_z[max_index]
        self.max_unc = self.unc_at_zero_crossings[max_index]


    def analyze_uncertainty(self, above_ground = True, z_0 = 0.0):
        """
        Gets the uncertainties at the zero-crossing points and finds the point with maximum uncertainty.

        Parameters
        ----------
            above_ground (bool): If True, finds the maximum uncertainty where the z value is greater than z_0.
            z_0 (float): Maximum uncertainty points need to be above z_0
        """
        self.get_uncertainties_at_zero_crossings()
        self.find_max_uncertainty_coordinates(above_ground, z_0)


    def update_data_and_model(self, points, normals, d_outside = 0.04, d_inside = 0.04, above_ground = False, z_0 = 0.0):
        """
        Update the GP regressor model with a new point or points for decreasing the uncertainty at that point/region.

        Parameters
        ----------
            points (ndarray or list): New point or array of new points to add to the training data. Each point consists of the coordinates (x, y, z).
            normals (ndarray or list): Normal vector of the new point or array of normals of the new points. Consists of (nx, ny, nz).
            d_outside (float): Step size variable for initializing points outside of the surface.
            d_inside (float): Step size variable for initializing points inside of the surface.
            above_ground (bool): If True, finds the maximum uncertainty where the z value is greater than z_0.
            z_0 (float): Maximum uncertainty points need to be above z_0
        """
        # Convert to numpy arrays if inputs are lists
        points = np.asarray(points)
        normals = np.asarray(normals)

        # Check if points and normals are 2D arrays
        if points.ndim == 1:
            points = np.asarray([points])  # Convert single point to a ndarray of points
        if normals.ndim == 1:
            normals = np.asarray([normals])  # Convert single normal to a ndarray of normals

        # Follow the normal vector to create training data outside the original surface:
        points_out = points + d_outside * normals
        fminus = -1 * np.ones(points.shape[0]) * d_outside  # assign y(x) = -1 to the points outside the surface

        # Concatenate the sub-parts to create the training data:
        self.X_train = np.vstack((self.X_train, points, points_out))
        self.y_train = np.hstack((self.y_train, np.zeros(points.shape[0]), fminus))

        # Fit GP model to the new training data and predict mean and covariance at the evaluation points
        self.gp_regressor.fit(self.X_train, self.y_train)
        self.mu_s, self.cov_s = self.gp_regressor.predict(self.Xstar, return_cov=True)

        self.analyze_uncertainty(above_ground, z_0)