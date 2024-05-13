import open3d as o3d
from gp.gpis import GPIS

class sense:
    def __init__(self):
        self.points = []
        self.normals = []
        self.timesteps = []

        self.nr_points_added = 0

        self.gpis = GPIS(obj_translate = [-0.3, 0, 0.735], d_outside = 0.04, d_inside = 0.04, resolution = 20)
        self.created_gp_model = False
        self.nr_points_to_create_gp_model = 0

    
    def add_point(self, x, y, z, t, n_x = 0.0, n_y = 0.0, n_z = 0.0):
        """
        Add single point from force torque sensor.
        x, y, z are point coordinates.
        t is timestep to match the point.
        """
        self.points.append([x, y, z])
        self.timesteps.append(t)
        self.normals.append([n_x, n_y, n_z])
        self.nr_points_added += 1
    
    
    def create_gp_model(self):
        self.gpis.create_gp_model(self.points, self.normals)
        self.nr_points_to_create_gp_model = self.return_nr_points()
        self.created_gp_model = True


    def update_gp_model(self):
        points = self.points[self.nr_points_to_create_gp_model:]
        normals = self.normals[self.nr_points_to_create_gp_model:]
        self.gpis.update_gp_model(points, normals)
        self.nr_points_to_create_gp_model = self.return_nr_points()


    def vizualise_pcd(self):
        pointSet = o3d.geometry.PointCloud()
        pointSet.points = o3d.utility.Vector3dVector(self.points)
        o3d.visualization.draw_geometries([pointSet])

    
    def visualize_gp_surface(self): # Visualize the predicted surface
        self.gpis.visu_surface()


if __name__ == "__main__":
    test = sense()
    test.add_point(0, 0, 0, 0)
    test.add_point(1, 1, 1, 1)
    test.add_point(2, 2, 2, 2)
    test.add_point(3, 3, 3, 3)
    test.vizualise_pcd() # Test to see if we get a line with 4 points
