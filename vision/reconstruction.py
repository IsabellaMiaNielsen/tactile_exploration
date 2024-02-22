import open3d as o3d

class sense:
    def __init__(self):
        self.points = []
        self.timesteps = []

    
    def add_point(self, x, y, z, t):
        """
        Add single point from force torque sensor.
        x, y, z are point coordinates.
        t is timestep to match the point.
        """
        self.points.append([x, y, z])
        self.timesteps.append(t)


    def vizualise_pcd(self):
        pointSet = o3d.geometry.PointCloud()
        pointSet.points = o3d.utility.Vector3dVector(self.points)
        o3d.visualization.draw_geometries([pointSet])


if __name__ == "__main__":
    test = sense()
    test.add_point(0, 0, 0, 0)
    test.add_point(1, 1, 1, 1)
    test.add_point(2, 2, 2, 2)
    test.add_point(3, 3, 3, 3)
    test.vizualise_pcd() # Test to see if we get a line with 4 points
