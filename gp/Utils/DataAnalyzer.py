import numpy as np
from scipy.optimize import brentq
from Utils.Visualizer import Visualizer as Visu

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


    def find_max_uncertainty_coordinates(self, above_ground = False):
        """
        Finds the coordinates (x,y,z) corresponding to the maximum uncertainty estimated by the GP regressor.
        
        Parameters
        ----------
        above_ground (bool): If True, finds the maximum uncertainty where the z value is greater than 0.0.
        """
        if above_ground:
            # Filter out zero-crossing points with z value <= 0.0
            filtered_indices = np.where(self.zero_crossings_z > 0.0)[0]
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


    def analyze_uncertainty(self, plot_uncertainties = False, above_ground = False):
        """
        Plots the uncertainties at the zero-crossing points and finds the point with maximum uncertainty.

        Parameters
        ----------
        plot_uncertainties (bool): Flag indicating whether to visualize uncertainties after updating the model.
        above_ground (bool): If True, finds the maximum uncertainty where the z value is greater than 0.0.
        """
        self.get_uncertainties_at_zero_crossings()
        self.find_max_uncertainty_coordinates(above_ground)

        print("Coordinates of maximum uncertainty point: ({:.4f}, {:.4f}, {:.4f})".format(self.max_x, self.max_y, self.max_z))
        print("Uncertainty: {:.5f}".format(self.max_unc))

        if plot_uncertainties:
            Visu.plot_uncertainties_3D(self.zero_crossings_x, self.zero_crossings_y, self.zero_crossings_z, self.unc_at_zero_crossings)


    def update_with_new_point(self, new_p, plot_uncertainties = False, above_ground = False):
        """
        Update the GP regressor model with a new point or points for decreasing the uncertainty at that point/region.

        Parameters
        ----------
        new_p (ndarray): New point or array of new points to add to the training data. Each point consists of the coordinates (x, y, z) and the normal vector (nx, ny, nz).
        plot_uncertainties (bool): Flag indicating whether to visualize uncertainties after updating the model.
        above_ground (bool): If True, finds the maximum uncertainty where the z value is greater than 0.0.
        """
        d_neg = 1
        # d_pos = 1

        if new_p.ndim == 1:
            # Follow the normal vector to create training data outside the original surface:
            points_out = new_p[:3] + d_neg * new_p[3:6]
            points_out2 = new_p[:3] + 2 * d_neg * new_p[3:6]
            fminus = -1 * d_neg  # assign y(x) = -1 to the points outside the surface

            # Follow the normal vector to create training data inside the original surface:
            # points_in = new_p[:3] - d_pos * new_p[3:6]
            # fone = d_pos  # assign y(x) = +1 to the points inside the surface

            # Concatenate the sub-parts to create the training data:
            self.X_train = np.vstack((self.X_train, new_p[:3], points_out, points_out2))#, points_in))
            self.y_train = np.hstack((self.y_train, [0.], fminus, fminus))#, fone))
        else:
            # Follow the normal vector to create training data outside the original surface:
            points_out = new_p[:, :3] + d_neg * new_p[:, 3:6]
            points_out2 = new_p[:, :3] + 2 * d_neg * new_p[:, 3:6]
            fminus = -1 * np.ones(new_p.shape[0]) * d_neg  # assign y(x) = -1 to the points outside the surface

            # points_in = new_p[:, :3] - d_pos * new_p[:, 3:6]
            # fone = np.ones(len(points_in)) * d_pos  # assign y(x) = +1 to the points inside the surface

            # Concatenate the sub-parts to create the training data:
            self.X_train = np.vstack((self.X_train, [arr[:3] for arr in new_p], points_out, points_out2))#, points_in))
            self.y_train = np.hstack((self.y_train, np.zeros(new_p.shape[0]), fminus, fminus))#, fone))


        # Fit GP model to the new training data and predict mean and covariance at the evaluation points
        self.gp_regressor.fit(self.X_train, self.y_train)
        self.mu_s, self.cov_s = self.gp_regressor.predict(self.Xstar, return_cov=True)

        # Visualization.plot_surface(self.Xstar, self.mu_s, self.cov_s)

        self.analyze_uncertainty(plot_uncertainties, above_ground)