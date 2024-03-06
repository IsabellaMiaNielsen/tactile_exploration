import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from mayavi import mlab
import open3d


######################################################################################################
# DataLoader Class for 
#       - loading the point cloud data and creating the training data
#       - generating the sample points for the GP regressor
#       - defining the evaluation limits
# Includes the following functions
#       - load_and_create_train_data
#       - generate_Xstar
#       - min_max_bunny
#       - min_max_rectangle
######################################################################################################
class DataLoader:
    @staticmethod
    def load_and_create_train_data(path, d_pos = 1, d_neg = 1):
        """
        Loads the point cloud data from a file saved in path.
        Each row in the file contains one point in the format:
            x y z n_x n_y n_z
        where n_x, n_y and n_z are the components of the normal vector.

        Parameters
        ----------
        path (str): The file path to the point cloud file
        d_pos (float or int): Used as step size variable to initialize points inside of the surface
        d_neg (float or int): Used as step size variable to initialize points outside of the surface

        Returns
        -------
        X_train (ndarray): Training data consists of points outside, inside and on the surface
        y_train (ndarray): Refers to the values of the implicit function.
            Assigns y(x) = +1 to points inside the surface
            Assigns y(x) = -1 to points outside the surface
            Assigns y(x) = 0 to points on the surface
        point_cloud (ndarray): Point cloud consisting of only the x, y and z position of the points
        """
        point_list = np.loadtxt(path)
        point_cloud = point_list[:, :3]

        # Follow the normal vector to create training data outside the original surface:
        points_out = point_list[:, :3] + d_neg * point_list[:, 3:6]
        points_out2 = point_list[:, :3] + 2 * d_neg * point_list[:, 3:6]


        # Defining points underneath the object (z < 0 means inside the table, where the object is placed on)
        # Define the range for x and y
        min_x, max_x = -2, 12
        min_y, max_y = -2, 7

        # Define the number of points along x and y
        num_points_x, num_points_y = 8, 8

        # Generate linearly spaced points for x and y within the range
        x = np.linspace(min_x, max_x, num_points_x)
        y = np.linspace(min_y, max_y, num_points_y)

        # Create 2D grids for x and y
        X, Y = np.meshgrid(x, y)
        # Define the constant value for z
        Z = -1 * np.ones_like(X)

        # Stack the x, y, and z arrays together
        points_below = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        fminus_below = -1 * np.ones(len(points_below)) * d_neg



        # # Defining points above the object (z > 10 means above the bounding box that limits the size of the object)
        # # Define the range for x and y
        # min_x, max_x = -2, 12
        # min_y, max_y = -2, 7

        # # Define the number of points along x and y
        # num_points_x, num_points_y = 8, 8

        # # Generate linearly spaced points for x and y within the range
        # x = np.linspace(min_x, max_x, num_points_x)
        # y = np.linspace(min_y, max_y, num_points_y)

        # # Create 2D grids for x and y
        # X, Y = np.meshgrid(x, y)
        # # Define the constant value for z
        # Z = 10 * np.ones_like(X)

        # # Stack the x, y, and z arrays together
        # points_above = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        # fminus_above = -1 * np.ones(len(points_above)) * d_neg




        # Follow the normal vector to create training data inside the original surface:
        points_in = point_list[:, :3] - d_pos * point_list[:, 3:6]

        fone = np.ones(len(points_in)) * d_pos  # assign y(x) = +1 to the points inside the surface
        fminus = -1 * np.ones(len(points_out)) * d_neg  # assign y(x) = -1 to the points outside the surface
        fzero = np.zeros(len(point_cloud))  # assign y(x) = 0 to the points on the surface

        # Concatenate the sub-parts to create the training data:
        X_train = np.vstack((point_cloud, points_in, points_out, points_out2, points_below))
        y_train = np.hstack((fzero, fone, fminus, fminus, fminus_below))  # flattening is required to avoid errors
        return X_train, y_train, point_cloud

    @staticmethod
    def generate_Xstar(minx, maxx, miny, maxy, minz, maxz, resolution):
        """
        Generates Xstar based on min max and resolution.
        Xstar is used for Gaussian Process prediction to get the mean and covariance of the estimated surface.
        
        Parameters
        ----------
        min, max (float64): Defines the evaluation limits of the implicit funciton predicted by the GP Regressor
        resolution (int): Distance between the sample points. resolution > 20 leads to RAM problems

        Returns
        -------
        Xstar (ndarray): Refers to the grid of points between the min and max evaluation limits
        """
        x = np.linspace(minx, maxx, resolution)
        y = np.linspace(miny, maxy, resolution)
        z = np.linspace(minz, maxz, resolution)

        X, Y, Z = np.meshgrid(x, y, z)
        Xstar = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

        desired_size = int((Xstar.shape[0])**(1/3)) + 1
        Xstar = np.pad(Xstar, ((0, desired_size**3 - Xstar.shape[0]), (0, 0)), mode='constant')
        return Xstar

    @staticmethod
    def min_max_bunny(X_train):
        """
        Defines min and max evaluation limits for the bunny point cloud.
        """
        minx, maxx = np.min(X_train[:, 0]), np.max(X_train[:, 0])
        miny, maxy = np.min(X_train[:, 1]) - 0.3, np.max(X_train[:, 1]) + 0.3
        minz, maxz = np.min(X_train[:, 2]), np.max(X_train[:, 2])
        return minx, maxx, miny, maxy, minz, maxz

    @staticmethod
    def min_max_rectangle():
        """
        Defines min and max evaluation limits for the rectangle point cloud. 
        """
        minx, maxx = -2, 12
        miny, maxy = -2, 6
        minz, maxz = 0, 7
        return minx, maxx, miny, maxy, minz, maxz



######################################################################################################
# Visualization Class for all visualization functions
# Includes the following functions
#       - plot_point_cloud
#       - plot_point_cloud_open3d
#       - plot_train_point_cloud
#       - plot_gp_mean
#       - plot_surface
#       - plot_uncertainties_2D
#       - plot_uncertainties_3D
######################################################################################################
class Visualization:
    @staticmethod
    def plot_point_cloud(point_cloud):
        """
        Plots the points saved in the point cloud in a scatter plot.

        Parameters
        ----------
        point_cloud (ndarray): Point cloud consisting of only the x, y and z position of the points
        """
        # Plot the point cloud
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Point cloud")
        plt.show()

    @staticmethod
    def plot_point_cloud_open3d(point_cloud, Xstar):
        """
        Plots the point cloud with the maximum uncertainty point highlighted.

        Parameters
        ----------
        point_cloud (ndarray): Original point cloud data
        Xstar (ndarray): Grid of points between the evaluation limits
        """
        original_pcd = open3d.geometry.PointCloud()
        original_pcd.points = open3d.utility.Vector3dVector(point_cloud)
        original_pcd.paint_uniform_color([0, 0, 1])  # entirely blue

        testpcd = open3d.geometry.PointCloud()
        testpcd.points = open3d.utility.Vector3dVector(Xstar)
        testpcd.paint_uniform_color([1, 0, 0])

        open3d.visualization.draw_geometries([testpcd, original_pcd])

    @staticmethod
    def plot_train_point_cloud(X_train, y_train):
        """
        Plots the points saved in the training point cloud X_train as a scatter plot.

        Parameters
        ----------
        X_train (ndarray): Point cloud consisting of only the x, y and z position of the points
        y_train (ndarray): Array containing the labels (-1, 0, 1) for each point in the point cloud
        """
        # Define colors based on y_train values
        colors = ['r' if label == min(y_train) else 'b' if label == 0 else 'y' for label in y_train]

        # Plot the point cloud with specified colors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=colors, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Training point cloud")

        # Create a legend
        legend_labels = {'r': 'Outside', 'b': 'On the surface', 'y': 'Inside'}
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label) for color, label in legend_labels.items()]
        ax.legend(handles=legend_handles, loc='best', fontsize='x-small')
        plt.show()

    @staticmethod
    def plot_gp_mean(Xstar, mu_s, min_surface = -0.001, max_surface = 0.001):
        """
        Plots the mean predicted by the GP regressor for each point in Xstar.

        Parameters
        ----------
        Xstar (ndarray): Refers to the grid of points between the evaluation limits for which the GP regressor predicts the surface
        mu_s (ndarray): Array containing the predicted mean by the GP regressor for each point in Xstar
        min_surface, max_surface (float): Min and max predicted mean value to still be considered as a point on the surface
        """
        # Define colors based on y_train values
        colors = ['r' if mean < min_surface else 'b' if (mean > min_surface and mean < max_surface) else 'y' for mean in mu_s]

        # Plot the point cloud with specified colors
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xstar[:, 0], Xstar[:, 1], Xstar[:, 2], c=colors, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Mean predicted by GP regressor")
        plt.show()

    @staticmethod
    def plot_surface(Xstar, mu_s, cov_s, plot_uncertainty = False):
        """
        Plots the surface of the object that has beed estimated by the GP model.

        Parameters
        ----------
        Xstar (ndarray): Refers to the grid of points between the evaluation limits for which the GP regressor predicts the surface
        mu_s (ndarray): Array containing the predicted mean by the GP regressor for each point in Xstar
        cov_s (ndarray): Array containing the covariance estimated by the GP regressor for each point in Xstar
        plot_uncertainty (bool): Flag indicating wether to visualize each uncertainty level with a separate contour or to only visualize the surface contour
        """
        tsize=int((Xstar.shape[0])**(1/3)) + 1
        xeva = Xstar.T[0, :].reshape((tsize,tsize,tsize))
        means_reshaped = mu_s.reshape(xeva.shape)

        mlab.clf()
        mlab.contour3d(means_reshaped, contours=[-0.001, 0.0, 0.001])

        if plot_uncertainty:
            uncertainty = np.diag(cov_s)
            uncertainty_reshaped = uncertainty.reshape(xeva.shape)
            min_contour = np.min(uncertainty)
            max_contour = np.max(uncertainty)
            mlab.contour3d(uncertainty_reshaped, contours=list(np.linspace(min_contour, max_contour, 10))) # Each contour represents one uncertainty level
        mlab.show()

    @staticmethod
    def plot_uncertainties_2D(uncertainties_at_zero_crossings):
        """
        Plots the uncertainties at the estimated surface locations in a 2D plot.

        Parameters
        ----------
        uncertainties_at_zero_crossings (ndarray): Uncertainties at the zero-crossing points
        """
        # Plot only the uncertainties at the zero-crossing points
        plt.plot(uncertainties_at_zero_crossings)
        plt.xlabel('Zero-crossing Index')
        plt.ylabel('Uncertainty')
        plt.title('Uncertainty at zero-crossing points')
        plt.show()

    @staticmethod
    def plot_uncertainties_3D(x_coords, y_coords, z_coords, uncertainties_at_zero_crossings):
        """
        Plots uncertainties at the zero-crossing points.

        Parameters
        ----------
        x_coords (ndarray): X coordinates of the zero-crossing points
        y_coords (ndarray): Y coordinates of the zero-crossing points
        z_coords (ndarray): Z coordinates of the zero-crossing points
        uncertainties_at_zero_crossings (ndarray): Uncertainties at the zero-crossing points
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_coords, y_coords, z_coords, c=uncertainties_at_zero_crossings, cmap='viridis', s=20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Uncertainties at zero-crossing points')
        plt.show()
    


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
            Visualization.plot_uncertainties_3D(self.zero_crossings_x, self.zero_crossings_y, self.zero_crossings_z, self.unc_at_zero_crossings)


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



#########################
######### main  #########
#########################
def main():
    # Load and create train data
    # X_train, y_train, point_cloud = DataLoader.load_and_create_train_data('Pointclouds/bunny.txt', d_pos = 0.2, d_neg = 0.2)
    X_train, y_train, point_cloud = DataLoader.load_and_create_train_data('Pointclouds/rectangle.txt', d_pos = 1, d_neg = 1)

    # Plot the point cloud
    # Visualization.plot_point_cloud(point_cloud)
    # Visualization.plot_train_point_cloud(X_train, y_train)

    # Evaluation limits:
    # minx, maxx, miny, maxy, minz, maxz = DataLoader.min_max_bunny(X_train)
    minx, maxx, miny, maxy, minz, maxz = DataLoader.min_max_rectangle()

    resolution = 20  # grid resolution for evaluation (my computer can handle a max of 20 without changing any RAM settings)

    # Generate Xstar
    Xstar = DataLoader.generate_Xstar(minx, maxx, miny, maxy, minz, maxz, resolution)

    # Plot all sample points
    # Visualization.plot_point_cloud(Xstar)
    # Visualization.plot_point_cloud_open3d(point_cloud, Xstar)

    # Define the Gaussian Process Regressor model
    kernel = 1.0 * RBF(length_scale=1.0)
    noise_3D = 0.1
    gp_regressor = GP_Regressor(kernel=kernel, alpha=noise_3D**2)

    # Fit GP model to training data
    gp_regressor.fit(X_train, y_train)

    # Predict mean and covariance at evaluation points
    mu_s, cov_s = gp_regressor.predict(Xstar, return_cov=True)

    # Plots the mean predicted by the GP regressor for each point in Xstar
    # Visualization.plot_gp_mean(Xstar, mu_s)
    # Visualization.plot_surface(Xstar, mu_s, cov_s)

    dataAnalyzer = DataAnalyzer(X_train, y_train, Xstar, gp_regressor, mu_s, cov_s)
    dataAnalyzer.analyze_uncertainty()


    #####################
    # Adding new points #
    #####################

    new_points = [
        np.array([9.8, 4.0, 4.8, 0.0, 1.0, 0.0]),
        np.array([[0.1, 0.1, 5.0, 0.0, 0.0, 1.0], [0.2, 0.2, 5.0, 0.0, 0.0, 1.0], [0.2, 0.0, 4.8, 0.0, -1.0, 0.0],
                  [0.01, 0.01, 5.0, 0.0, 0.0, 1.0], [0.1, 1, 5.0, 0.0, 0.0, 1.0], [1, 0.1, 5.0, 0.0, 0.0, 1.0]]),
        np.array([4.0, 4.0, 0.5, 0.0, 1.0, 0.0]),
        np.array([8.3, 4.0, 4.4, 0.0, 1.0, 0.0]),
        np.array([5.4, 4.0, 0.7, 0.0, 1.0, 0.0]),
        np.array([3.2, 4.0, 0.7, 0.0, 1.0, 0.0]),
        np.array([[5.3, 4.0, 4.8, 0.0, 1.0, 0.0], [5.1, 4.0, 4.9, 0.0, 1.0, 0.0]]),
        np.array([[5.3, 3.5, 5.0, 0.0, 0.0, 1.0], [5.5, 3.6, 5.0, 0.0, 0.0, 1.0], [5.3, 3.8, 5.0, 0.0, 0.0, 1.0], [5.1, 3.8, 5.0, 0.0, 0.0, 1.0], [5.1, 3.4, 5.0, 0.0, 0.0, 1.0], [5.4, 3.4, 5.0, 0.0, 0.0, 1.0], [5.3, 3.95, 5.0, 0.0, 0.0, 1.0]]),

        np.array([6.0, 0.0, 4.9, 0.0, -1.0, 0.0]),
        np.array([6.0, 0.0, 4.9, 0.0, -1.0, 0.0]),
    ]

    for point in new_points:
        dataAnalyzer.update_with_new_point(point, above_ground = True)

    new_p = np.array([3.0, 3.5, 5.0, 0.0, 0.0, 1.0])
    dataAnalyzer.update_with_new_point(new_p, plot_uncertainties = True, above_ground = True)

    Visualization.plot_surface(dataAnalyzer.Xstar, dataAnalyzer.mu_s, dataAnalyzer.cov_s)
    print("Total number of points measured: ", np.count_nonzero(dataAnalyzer.y_train == 0))

        
if __name__ == "__main__":
    main()