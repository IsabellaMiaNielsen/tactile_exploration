import matplotlib.pyplot as plt
import numpy as np
import open3d
from mayavi import mlab


######################################################################################################
# Visualizer Class for all visualization functions
# Includes the following functions
#       - plot_point_cloud
#       - plot_point_cloud_open3d
#       - plot_point_cloud_xstar_open3d
#       - plot_train_point_cloud
#       - plot_gp_mean
#       - plot_surface
#       - plot_uncertainties_2D
#       - plot_uncertainties_3D
######################################################################################################
class Visualizer:
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
    def plot_point_cloud_open3d(point_cloud):
        """
        Plots the point cloud including the world frame axes in open3d.

        Parameters
        ----------
            point_cloud (ndarray): Original point cloud data
        """
        # Create axes
        axes = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])


        # frame_pos = [-0.3, 0, 0.735]
        # frame_orientation = [0, -1.57, -1.57]
        
        # # Rotate the default coordinate frame to match the specified orientation
        # rotation_matrix = open3d.geometry.get_rotation_matrix_from_xyz(frame_orientation)

        # # Apply translation to the rotated coordinate frame
        # center_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]).translate(np.array(frame_pos))
        # center_frame = center_frame.rotate(rotation_matrix)
    

        # Visualize the point cloud, axes, and point of interest sphere
        open3d.visualization.draw_geometries([point_cloud, axes]) #, center_frame])


    @staticmethod
    def plot_point_cloud_xstar_open3d(point_cloud, Xstar):
        """
        Plots the point cloud including the grid of points Xstar in open3d.

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
    def plot_gp_mean(Xstar, mu_s, min_surface = -0.01, max_surface = 0.01):
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