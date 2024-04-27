import numpy as np
from sklearn.gaussian_process.kernels import RBF
from utils.data_loader import *
from utils.gp_regressor import *
from utils.visualizer import Visualizer as Visu
from utils.data_analyzer import *


#########################
######### main ##########
#########################
def main():
    # Load and create train data
    # X_train, y_train, point_cloud = DataLoader.load_and_create_train_data('pointclouds/bunny.txt', d_pos = 0.2, d_neg = 0.2)
    X_train, y_train, point_cloud = DataLoader.load_and_create_train_data('gp/pointclouds/rectangle.txt', d_pos = 1, d_neg = 1)

    # Plot the point cloud
    # Visu.plot_point_cloud(point_cloud)
    # Visu.plot_train_point_cloud(X_train, y_train)

    # Evaluation limits:
    # minx, maxx, miny, maxy, minz, maxz = DataLoader.min_max_bunny(X_train)
    minx, maxx, miny, maxy, minz, maxz = DataLoader.min_max_rectangle()

    resolution = 20  # grid resolution for evaluation (my computer can handle a max of 20 without changing any RAM settings)

    # Generate Xstar
    Xstar = DataLoader.generate_Xstar(minx, maxx, miny, maxy, minz, maxz, resolution)

    # Plot all sample points
    # Visu.plot_point_cloud(Xstar)
    # Visu.plot_point_cloud_open3d(point_cloud)
    # Visu.plot_point_cloud_xstar_open3d(point_cloud, Xstar)

    # Define the Gaussian Process Regressor model
    kernel = 1.0 * RBF(length_scale=1.0)
    noise_3D = 0.1
    gp_regressor = GP_Regressor(kernel=kernel, alpha=noise_3D**2)

    # Fit GP model to training data
    gp_regressor.fit(X_train, y_train)

    # Predict mean and covariance at evaluation points
    mu_s, cov_s = gp_regressor.predict(Xstar, return_cov=True)

    # Plots the mean predicted by the GP regressor for each point in Xstar
    # Visu.plot_gp_mean(Xstar, mu_s)
    Visu.plot_surface(Xstar, mu_s, cov_s)

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
        dataAnalyzer.update_with_new_point(point, plot_uncertainties = False, above_ground = True)

    new_p = np.array([3.0, 3.5, 5.0, 0.0, 0.0, 1.0])
    dataAnalyzer.update_with_new_point(new_p, plot_uncertainties = True, above_ground = True)

    Visu.plot_surface(dataAnalyzer.Xstar, dataAnalyzer.mu_s, dataAnalyzer.cov_s)
    print("Total number of points measured: ", np.count_nonzero(dataAnalyzer.y_train == 0))

        
if __name__ == "__main__":
    main()