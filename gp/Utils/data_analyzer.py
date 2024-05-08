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
#       - update_data_and_model
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
        self.zero_crossings = None # Coordinates of the zero-crossing points
        self.max_unc = None
        self.max_unc_pos = None


    def get_uncertainties_at_zero_crossings(self):
        """
        1. Reshapes the arrays for easy calculation.
        2. Computes the sign changes of mu_s (predicted mean by Gaussian Process) in all axes directions (x, y and z axis).
        3. If the sign changes, a zero-crossing occurs. 
        4. For better accuracy save the two indices between which the zero crossing occurs.
        5. Only take one index out of the two where the predicted mean (mu_s) is closer to zero which means closer to the zero-crossing.
        6. Finds the actual zero-crossing position and the uncertainties at the zero-crossing points based on the indices.
        """
        tsize=int((self.Xstar.shape[0])**(1/3)) + 1
        Xstar_reshape = self.Xstar.reshape((tsize,tsize,tsize,3))
        mu_s_reshaped = self.mu_s.reshape((tsize,tsize,tsize))
        unc_reshaped = self.unc.reshape((tsize,tsize,tsize))

        sign_mu_s = np.sign(self.mu_s)
        sign_mu_s = sign_mu_s.reshape((tsize,tsize,tsize))

        # Calculate sign changes along each axis
        sign_changes = []

        for axis in range(3): # Iterate over 3 axes (3D space)
            sign_changes_axis = np.asarray(np.where(np.diff(sign_mu_s, axis=axis))).T
            for index in sign_changes_axis:
                current_index = list(index)
                other_index = list(index)
                other_index[axis] += 1 

                sign_changes.append([current_index, other_index])

        sign_changes = np.asarray(sign_changes)

        # Initialize array to store unique zero crossing indices
        unique_zero_crossing_indices = set()

        # Iterate through sign_changes
        for idx_pair in sign_changes:
            # Initialize variables to store closest index and minimum distance to zero
            closest_index = None
            min_distance = np.inf
            
            # Iterate through pairs of indices
            for idx in idx_pair:
                # Calculate distance to zero
                distance_to_zero = np.abs(mu_s_reshaped[tuple(idx)])
                
                # Update closest index if current index is closer to zero
                if distance_to_zero < min_distance:
                    closest_index = idx
                    min_distance = distance_to_zero
            
            # Add closest index to zero_crossing_indices
            unique_zero_crossing_indices.add(tuple(closest_index))

        # Convert unique_zero_crossing_indices to numpy array
        zero_crossing_indices = np.array(list(unique_zero_crossing_indices))

        # Get actual x y z values of the zero crossings
        self.zero_crossings = Xstar_reshape[zero_crossing_indices[:, 0], zero_crossing_indices[:, 1], zero_crossing_indices[:, 2]]

        # Get uncertainties based on the indices of zero-crossing points
        self.unc_at_zero_crossings = unc_reshaped[zero_crossing_indices[:, 0], zero_crossing_indices[:, 1], zero_crossing_indices[:, 2]]


    def find_max_uncertainty_coordinates(self, z_0 = 0.0):
        """
        Finds the coordinates (x,y,z) corresponding to the maximum uncertainty estimated by the GP regressor.
        
        Parameters
        ----------
            z_0 (float): Maximum uncertainty points need to be above z_0
        """
        # Filter out zero-crossing points with z value <= z_0
        filtered_indices = np.where(self.zero_crossings[:, 2] > z_0)[0]
        filtered_uncertainties = self.unc_at_zero_crossings[filtered_indices]

        # Find the index of the maximum uncertainty
        max_index = filtered_indices[np.argmax(filtered_uncertainties)]
       
        # Retrieve the x, y, and z coordinates corresponding to the maximum uncertainty
        self.max_unc_pos = self.zero_crossings[max_index]
        self.max_unc = self.unc_at_zero_crossings[max_index]


    def analyze_uncertainty(self, z_0 = 0.0):
        """
        Gets the uncertainties at the zero-crossing points and finds the point with maximum uncertainty.

        Parameters
        ----------
            z_0 (float): Maximum uncertainty points need to be above z_0
        """
        self.get_uncertainties_at_zero_crossings()
        self.find_max_uncertainty_coordinates(z_0)


    def update_data_and_model(self, points, normals, d_outside = 0.04, d_inside = 0.04, z_0 = 0.0):
        """
        Update the GP regressor model with a new point or points for decreasing the uncertainty at that point/region.

        Parameters
        ----------
            points (ndarray or list): New point or array of new points to add to the training data. Each point consists of the coordinates (x, y, z).
            normals (ndarray or list): Normal vector of the new point or array of normals of the new points. Consists of (nx, ny, nz).
            d_outside (float): Step size variable for initializing points outside of the surface.
            d_inside (float): Step size variable for initializing points inside of the surface.
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

        self.analyze_uncertainty(z_0)