import numpy as np

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