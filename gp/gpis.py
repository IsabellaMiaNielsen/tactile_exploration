import numpy as np
import open3d as o3d
from sklearn.gaussian_process.kernels import RBF
from utils.gp_regressor import *
from utils.visualizer import Visualizer as Visu
from utils.data_analyzer import *

from utils.point_cloud import Point_cloud


######################################################################################################
# DataAnalyzer Class for 
#       - using GPIS to get max uncertainty points
# Includes the following functions
#       - create_gp_model
#       - update_gp_model
#       - visu_point_cloud
#       - visu_point_cloud_open3d
#       - visu_train_data
#       - visu_surface
#       - visu_surface_grid
#       - visu_uncertainty
######################################################################################################
class GPIS:
    def __init__(self, obj_translate = [-0.3, 0, 0.735], d_outside = 0.04, d_inside = 0.04, resolution = 20):
        """
        Initializes a GPIS object.

        Parameters
        ----------
            obj_translate (list): Coordinates for translation.
            d_outside (float): Step size variable for initializing points outside of the surface.
            d_inside (float): Step size variable for initializing points inside of the surface.
            resolution (int): Grid resolution for evaluation. Values greater than 20 may lead to RAM problems.
        """
        self.point_cloud = Point_cloud( 
            obj_translate = obj_translate, 
            d_outside = d_outside, 
            d_inside = d_inside, 
            resolution = resolution
            )


    def create_gp_model(self, points, normals):
        """
        Creates and fits a Gaussian Process (GP) model to the given points and normals.

        Parameters
        ----------
            points (ndarray): Array of points to initialize the model.
            normals (ndarray): Array of normals for initialization.
        
        Returns
        -------
            max_unc_position (list): Coordinates of the point with maximum uncertainty.
            max_unc (float): Maximum uncertainty value.
        """
        # Add points to point cloud of class Point_cloud
        self.point_cloud.create_point_cloud(points, normals)

        # Create train data
        self.point_cloud.create_train_data()

        # Define evaluation limits
        self.point_cloud.min_max_rounded_cube()

        # Generate Xstar
        self.point_cloud.generate_Xstar() 

        # Define the Gaussian Process Regressor model
        kernel = 1.0 * RBF(length_scale=0.04) # Optimized for rounded_cube: 1.0 and 0.04
        noise_3D = 0.001 # 0.001
        gp_regressor = GP_Regressor(kernel=kernel, alpha=noise_3D**2)

        # Fit GP model to training data
        gp_regressor.fit(self.point_cloud.X_train, self.point_cloud.y_train)

        # Predict mean and covariance at evaluation points
        self.mu_s, self.cov_s = gp_regressor.predict(self.point_cloud.Xstar, return_cov=True)

        self.dataAnalyzer = DataAnalyzer(self.point_cloud.X_train, self.point_cloud.y_train, self.point_cloud.Xstar, gp_regressor, self.mu_s, self.cov_s)
        self.dataAnalyzer.analyze_uncertainty()

        print("Coordinates of maximum uncertainty point: ({:.4f}, {:.4f}, {:.4f})".format(self.dataAnalyzer.max_x, self.dataAnalyzer.max_y, self.dataAnalyzer.max_z))
        print("Uncertainty: {:.5f}".format(self.dataAnalyzer.max_unc))
        return [self.dataAnalyzer.max_x, self.dataAnalyzer.max_y, self.dataAnalyzer.max_z], self.dataAnalyzer.max_unc


    def update_gp_model(self, points, normals):
        """
        Updates the GP model with an additional point or multiple additional points and normals.

        Parameters
        ----------
            points (ndarray): Array of points to add to the model.
            normals (ndarray): Array of normals belonging to the new points.
        
        Returns
        -------
            max_unc_position (list): Coordinates of the point with maximum uncertainty.
            max_unc (float): Maximum uncertainty value.
        """
        # Update open3d pointcloud with new points and normals
        self.point_cloud.update_point_cloud(points, normals)
        
        # Update gp model with new points and normals and compute maximum uncertainty
        self.dataAnalyzer.update_data_and_model(points, normals, d_outside=self.point_cloud.d_outside, d_inside=self.point_cloud.d_inside, above_ground=True, z_0=self.point_cloud.obj_translate[2])

        print("Coordinates of maximum uncertainty point: ({:.4f}, {:.4f}, {:.4f})".format(self.dataAnalyzer.max_x, self.dataAnalyzer.max_y, self.dataAnalyzer.max_z))
        print("Uncertainty: {:.5f}".format(self.dataAnalyzer.max_unc))
        return [self.dataAnalyzer.max_x, self.dataAnalyzer.max_y, self.dataAnalyzer.max_z], self.dataAnalyzer.max_unc


    def visu_point_cloud(self):
        """
        Visualizes the point cloud.
        """
        Visu.plot_point_cloud(np.asarray(self.point_cloud.point_cloud.points))


    def visu_point_cloud_open3d(self):
        """
        Visualizes the point cloud and the world frame in Open3D.
        """
        Visu.plot_point_cloud_open3d(self.point_cloud.point_cloud)


    def visu_train_data(self):
        """
        Visualizes the training data.
        """
        Visu.plot_train_point_cloud(self.point_cloud.X_train, self.point_cloud.y_train)


    def visu_surface(self):
        """
        Visualizes the surface.
        """
        Visu.plot_surface(self.point_cloud.Xstar, self.mu_s, self.cov_s)


    def visu_surface_grid(self):
        """
        Visualizes the surface as a point grid based on Xstar.
        """
        Visu.plot_gp_mean(self.point_cloud.Xstar, self.mu_s)


    def visu_uncertainty(self):
        """
        Visualizes the uncertainty for all points on the surface (predicted by the gp regressor).
        """
        Visu.plot_uncertainties_3D(self.dataAnalyzer.zero_crossings_x, self.dataAnalyzer.zero_crossings_y, self.dataAnalyzer.zero_crossings_z, self.dataAnalyzer.unc_at_zero_crossings)


if __name__ == "__main__":
    gpis = GPIS(obj_translate = [-0.3, 0, 0.735], d_outside = 0.04, d_inside = 0.04, resolution = 20)

    # Create example points from object file and split into train and test points with normals
    pc = Point_cloud()
    pc.load_from_object_file(num_points=100)
    train_points = np.asarray(pc.point_cloud.points)[:80] 
    train_normals = np.asarray(pc.point_cloud.normals)[:80]
    test_points = np.asarray(pc.point_cloud.points)[80:]
    test_normals = np.asarray(pc.point_cloud.normals)[80:]

    gpis.create_gp_model(train_points, train_normals)

    # gpis.visu_point_cloud_open3d()

    gpis.update_gp_model(test_points, test_normals)

    gpis.visu_surface()