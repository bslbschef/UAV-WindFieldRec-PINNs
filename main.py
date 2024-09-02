import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model.PhysicsInformedNN import PhysicsInformedNN
from util.tools import *

if __name__ == "__main__":
    N_train = 97000  # sparsity: 0.14%
    # layers = [4, 100, 100, 100, 100, 4]
    layers = [4, 50, 50, 50, 50, 4]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv('./data/LiDAR_data_xyztuvwphitheta_horizontal_add_random_points.csv')
    df_all_points = df
    x_all_points, y_all_points, z_all_points, t_all_points, u_all_points, v_all_points, w_all_points, phi_all_points, theda_all_points = extract_and_round_data_all(df_all_points)

    # NT = x_lidar.shape[0]
    torch.autograd.set_detect_anomaly(True)
    model = PhysicsInformedNN(x_all_points, y_all_points, z_all_points, t_all_points, u_all_points, v_all_points, w_all_points, phi_all_points, theda_all_points, layers, device)
    mode = 'Training'

    if mode == 'Training':
        model.train(500000, "./result/")
    # elif mode == 'Test':
    #
    #     model.load_state_dict(torch.load('save/PINN_500000.pth'))
    #
    #     # Test Data
    #     rand_t = 98.5
    #     xmax, xmin = x.max(), x.min()
    #     ymax, ymin = y.max(), y.min()
    #     zmax, zmin = z.max(), z.min()
    #     x_interval, y_interval, z_interval = 10, 10, 7.5
    #
    #     # predict_XY(data, model, rand_t, zmin, zmax, z_interval)
    #     # predict_XZ(data, model, rand_t, ymin, ymax, y_interval)
    #     # predict_YZ(data, model, rand_t, xmin, xmax, x_interval)
    #
    #     rand_x, rand_y, rand_z = 200, 100, 172.5
    #     predict_filed_XY(data, model, rand_t, rand_z)
    #     predict_filed_YZ(data, model, rand_t, rand_x)
    #     predict_filed_XZ(data, model, rand_t, rand_y)
    #
    #     rand_y = 190
    #     plane_data = data[(data[:, 1] == rand_y) & (data[:, 3] == rand_t)][:, [0, 2, 4, 5, 6]]
    #     # print("Shape of plane_data:", plane_data.shape)
    #     # print(plane_data[0, 1])
    #
    #     x_star = plane_data[:, 0].reshape(-1, 1)
    #     z_star = plane_data[:, 1].reshape(-1, 1)
    #     u_star = plane_data[:, 2].reshape(-1, 1)
    #     v_star = plane_data[:, 3].reshape(-1, 1)
    #     w_star = plane_data[:, 4].reshape(-1, 1)
    #
    #     Nx = x_star.shape[0]
    #     # plane_data_angle = train_data[train_data[:, 3] == rand_t][:, [7, 8]]
    #     # phi_star = plane_data_angle[0, 0].reshape(-1, 1)
    #     # theda_star = plane_data_angle[0, 1].reshape(-1, 1)
    #     # # print(phi_star)
    #     # # print(theda_star)
    #     # phi_star = np.tile(phi_star, (Nx, 1))
    #     # theda_star = np.tile(theda_star, (Nx, 1))
    #
    #     speed_star = calc_v_magnitude(u_star, v_star, w_star)
    #
    #     t_fixed = np.array([rand_t]).reshape(-1, 1)
    #     y_fixed = np.array([rand_y]).reshape(-1, 1)
    #     y_star = np.tile(y_fixed, (Nx, 1))  # Nx * Nz x 1
    #     t_star = np.tile(t_fixed, (Nx, 1))  # Nx * Nz x 1
    #
    #     # Prediction
    #     u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
    #     speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)
    #     # Re_value = model.Re.item()  # Assuming re is a scalar tensor
    #
    #     # error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    #     # error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    #     # error_w = np.linalg.norm(w_star-w_pred,2)/np.linalg.norm(w_star,2)
    #     error_speed = np.linalg.norm(speed_star - speed_pred, 2) / np.linalg.norm(speed_star, 2)
    #     # print('Error u: %e' % (error_u))
    #     # print('Error v: %e' % (error_v))
    #     # print('Error w: %e' % (error_w))
    #     print('Error speed: %e' % (error_speed))
    #     #     if error_speed < error_min:
    #     #         error_min = error_speed
    #     #         xz_min = rand_y
    #
    #     # Predict for plotting
    #     xz_location = plane_data[:, :2]
    #     # plot_solution_XZ(xz_location, u_pred, 1)
    #     # plot_solution_XZ(xz_location, v_pred, 2)
    #     # plot_solution_XZ(xz_location, w_pred, 3)
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #
    #     vmin = min(speed_pred.min(), speed_star.min())
    #     vmax = max(speed_pred.max(), speed_star.max())
    #
    #     im1 = plot_solution_XZ(xz_location, speed_pred, ax1, vmin, vmax)
    #     ax1.set_title('Predicted Speed')
    #     im2 = plot_solution_XZ(xz_location, speed_star, ax2, vmin, vmax)
    #     ax2.set_title('True Speed')
    #
    #     fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical', label='Speed')
    #
    #     # plt.tight_layout()
    #     plt.show()
    #
    #     plot_solution_XZ_Single(xz_location, np.abs(speed_star - speed_pred), 4)
