import numpy as np
import os
import open3d as o3d
# from gp.utils.visualizer import Visualizer as Visu
from utils.visualizer import Visualizer as Visu


######################################################################################################
# Class for working with point clouds.
# Includes the following functions
#       - create_point_cloud
#       - update_point_cloud
#       - load_from_object_file
#       - save_point_cloud
#       - load_point_cloud
#       - get_dimensions
#       - create_train_data
#       - min_max_rounded_cube
#       - generate_Xstar
######################################################################################################
class Point_cloud:
    def __init__(
            self, 
            obj_translate = [-0.3, 0, 0.735], 
            d_outside = 0.04, 
            d_inside = 0.04,
            resolution = 20
            ):
        """
        Initializes a Point_cloud object.

        Parameters
        ----------
            obj_translate (list): Coordinates for translation.
            d_outside (float): Step size variable for initializing points outside of the surface.
            d_inside (float): Step size variable for initializing points inside of the surface.
            resolution (int): Distance between the sample points. resolution > 20 may lead to RAM problems
        """
        self.obj_translate = obj_translate
        self.d_outside = d_outside
        self.d_inside = d_inside
        self.resolution = resolution

        self.point_cloud = o3d.geometry.PointCloud()


    def create_point_cloud(self, points, normals):
        """
        Creates a point cloud from given points and normals.

        Parameters
        ----------
            points (ndarray): Array of points.
            normals (ndarray): Array of normals.
        """
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.normals = o3d.utility.Vector3dVector(normals)


    def update_point_cloud(self, points, normals):
        """
        Updates the point cloud with additional points and normals.

        Parameters
        ----------
            points (ndarray or list): Array of points.
            normals (ndarray or list): Array of normals.
        """
        # Convert to numpy arrays if inputs are lists
        points = np.asarray(points)
        normals = np.asarray(normals)

        # Check if points and normals are 2D arrays
        if points.ndim == 1:
            points = np.asarray([points])  # Convert single point to a ndarray of points
        if normals.ndim == 1:
            normals = np.asarray([normals])  # Convert single normal to a ndarray of normals

        new_points_o3d = o3d.utility.Vector3dVector(points)
        new_normals_o3d = o3d.utility.Vector3dVector(normals)

        self.point_cloud.points.extend(new_points_o3d)
        self.point_cloud.normals.extend(new_normals_o3d)

    
    def load_from_object_file(self, stl_file_name = "rounded_cube.stl", num_points = 100):
        """
        Loads a point cloud from an STL file.

        Parameters
        ----------
            stl_file_name (str): Name of the STL file.
            num_points (int): Number of points to sample from the mesh.
        """
        # Directory to the assets folder
        assets_dir = "/scene_files/assets/"

        # Get the directory path of the current Python script
        self.script_dir = os.path.dirname(__file__)

        # Navigate to the parent directory of the script
        parent_dir = os.path.dirname(os.path.dirname(self.script_dir))

        # Load OBJ file
        mesh = o3d.io.read_triangle_mesh(parent_dir + assets_dir + stl_file_name)

        # Convert to point cloud
        self.point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)

        # Position of rounded cube -0.3 0 0.735
        self.point_cloud.translate(self.obj_translate)

        # Estimate surface normals pointing away from the object surface (not inside)
        self.point_cloud.estimate_normals()


    def save_point_cloud(self, filename="point_cloud.ply"):
        """
        Saves the point cloud to a file.

        Parameters
        ----------
            filename (str): File name of the desired point cloud (of type .ply)
        """
        # Save the point cloud to a file
        o3d.io.write_point_cloud(os.path.dirname(self.script_dir) + "/pointclouds/" + filename, self.point_cloud)
        

    def load_point_cloud(self, filename="point_cloud.ply"):
        """
        Loads the point cloud from a file.

        Parameters
        ----------
            filename (str): File name of the desired point cloud (of type .ply)
        """
        self.point_cloud = o3d.io.read_point_cloud(os.path.dirname(self.script_dir) + "/pointclouds/" + filename)


    def get_dimensions(self):
        """
        Computes the dimensions of the object.
        """
        # Compute the axis-aligned bounding box
        axis_aligned_bounding_box = self.point_cloud.get_axis_aligned_bounding_box()

        # Get the dimensions of the bounding box
        dimensions = axis_aligned_bounding_box.get_extent()
        print("Dimensions of the object (width, height, depth):", dimensions)


    def create_train_data(self):
        """
        Creates training data for the Gaussian Process.

        Assigns y(x) = +1 to points inside the surface
        Assigns y(x) = -1 to points outside the surface
        Assigns y(x) = 0 to points on the surface
        """
        # Follow the normal vector to create training data outside the original surface:
        points_out = np.asarray(self.point_cloud.points) + self.d_outside * np.asarray(self.point_cloud.normals)

        # Follow the normal vector to create training data inside the original surface:
        points_in = np.asarray(self.point_cloud.points) - self.d_inside * np.asarray(self.point_cloud.normals)

        fone = 1 * np.ones(len(points_in)) *self. d_inside  # assign y(x) = +1 to the points inside the surface
        fminus = -1 * np.ones(len(points_out)) * self.d_outside  # assign y(x) = -1 to the points outside the surface
        fzero = np.zeros(len(np.asarray(self.point_cloud.points)))  # assign y(x) = 0 to the points on the surface

        # Concatenate the sub-parts to create the training data:
        self.X_train = np.vstack((np.asarray(self.point_cloud.points), points_in, points_out))
        self.y_train = np.hstack((fzero, fone, fminus))


    def min_max_rounded_cube(self):
        """
        Defines min and max evaluation limits for the rounded cube point cloud. 
        """
        # self.minx, self.maxx = -0.5, 0.2
        # self.miny, self.maxy = -0.2, 0.5
        # self.minz, self.maxz = -0.2, 0.5
        self.minx, self.maxx = -0.25, 0.05 
        self.miny, self.maxy = -0.05, 0.25
        self.minz, self.maxz = -0.05, 0.25

        # if point cloud is translated, translate also the evaluation limits
        self.minx, self.maxx = self.minx + self.obj_translate[0], self.maxx + self.obj_translate[0]
        self.miny, self.maxy = self.miny + self.obj_translate[1], self.maxy + self.obj_translate[1]
        self.minz, self.maxz = self.minz + self.obj_translate[2], self.maxz + self.obj_translate[2]
    

    def generate_Xstar(self):
        """
        Generates Xstar that refers to the grid of points between the min and max evaluation limits
        Xstar is used for Gaussian Process prediction to get the mean and covariance of the estimated surface.
        """
        x = np.linspace(self.minx, self.maxx, self.resolution)
        y = np.linspace(self.miny, self.maxy, self.resolution)
        z = np.linspace(self.minz, self.maxz, self.resolution)

        X, Y, Z = np.meshgrid(x, y, z)
        self.Xstar = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

        desired_size = int((self.Xstar.shape[0])**(1/3)) + 1
        self.Xstar = np.pad(self.Xstar, ((0, desired_size**3 - self.Xstar.shape[0]), (0, 0)), mode='constant')
        

if __name__ == "__main__":
    point_cloud = Point_cloud()
    Visu.plot_point_cloud_open3d(point_cloud.point_cloud)
    point_cloud.get_dimensions()