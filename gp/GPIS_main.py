import numpy as np
from sklearn.gaussian_process.kernels import RBF
from utils.gp_regressor import *
from utils.visualizer import Visualizer as Visu
from utils.data_analyzer import *

from utils.point_cloud import Point_cloud

#########################
######### main ##########
#########################
def main():
    # Load and create train data
    point_cloud = Point_cloud(
            stl_file_name = "rounded_cube.stl", 
            num_points = 40, 
            center = [-0.3, 0, 0.735], 
            d_outside = 0.04, 
            d_inside = 0.04, 
            resolution = 20, # grid resolution for evaluation (my computer can handle a max of 20 without changing any RAM settings)
            translate = False
            )
    # point_cloud.save_point_cloud()
    # point_cloud.load_point_cloud()

    # Create train data
    point_cloud.create_train_data()

    # Plot the point cloud
    # Visu.plot_point_cloud(np.asarray(point_cloud.point_cloud.points))
    # Visu.plot_train_point_cloud(point_cloud.X_train, point_cloud.y_train)

    # Define evaluation limits
    point_cloud.min_max_rounded_cube()

    # Generate Xstar
    point_cloud.generate_Xstar() 

    # Visu.plot_point_cloud(point_cloud.Xstar)
    # Visu.plot_point_cloud_open3d(point_cloud.point_cloud)
    # Visu.plot_point_cloud_xstar_open3d(np.asarray(point_cloud.point_cloud.points), point_cloud.Xstar)

    # Define the Gaussian Process Regressor model
    kernel = 1.0 * RBF(length_scale=0.04) # Optimized for rounded_cube: 1.0 and 0.04
    noise_3D = 0.001 # 0.001
    gp_regressor = GP_Regressor(kernel=kernel, alpha=noise_3D**2)

    # Fit GP model to training data
    gp_regressor.fit(point_cloud.X_train, point_cloud.y_train)

    # Predict mean and covariance at evaluation points
    mu_s, cov_s = gp_regressor.predict(point_cloud.Xstar, return_cov=True)

    # Plots the mean predicted by the GP regressor for each point in Xstar
    # Visu.plot_gp_mean(point_cloud.Xstar, mu_s)
    Visu.plot_surface(point_cloud.Xstar, mu_s, cov_s)

    dataAnalyzer = DataAnalyzer(point_cloud.X_train, point_cloud.y_train, point_cloud.Xstar, gp_regressor, mu_s, cov_s)
    dataAnalyzer.analyze_uncertainty()


    #####################
    # Adding new points #
    #####################
    # new_p = [-0.1, 0.1, 0.2]
    
    # dataAnalyzer.update_with_new_point(new_p, d_outside = 0.04, d_inside = 0.04, plot_uncertainties = True, above_ground = True)

    # Visu.plot_surface(dataAnalyzer.Xstar, dataAnalyzer.mu_s, dataAnalyzer.cov_s)
    # print("Total number of points measured: ", np.count_nonzero(dataAnalyzer.y_train == 0))


if __name__ == "__main__":
    main()