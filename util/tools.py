import numpy as np
import torch
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def calc_v_magnitude(u, v, w):
    return np.sqrt(u ** 2 + v ** 2 + w ** 2)


def plot_solution_XY_Single(X_star, u_star, index):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    plt.figure(index)
    plt.pcolor(X, Y, U_star, cmap='jet')
    plt.colorbar()
    plt.savefig('X-Y_diff.png')
    plt.show()


def plot_solution_XZ_Single(X_star, u_star, index):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    z = np.linspace(lb[1], ub[1], nn)
    X, Z = np.meshgrid(x, z)

    U_star = griddata(X_star, u_star.flatten(), (X, Z), method='cubic')

    plt.figure(index)
    plt.pcolor(X, Z, U_star, cmap='jet')
    plt.colorbar()
    plt.savefig('X-Z_diff.png')
    plt.show()


def plot_solution_YZ_Single(X_star, u_star, index):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    y = np.linspace(lb[0], ub[0], nn)
    z = np.linspace(lb[1], ub[1], nn)
    Y, Z = np.meshgrid(y, z)

    U_star = griddata(X_star, u_star.flatten(), (Y, Z), method='cubic')

    plt.figure(index)
    plt.pcolor(Y, Z, U_star, cmap='jet')
    plt.colorbar()
    plt.savefig('Y-Z_diff.png')
    plt.show()


def plot_solution_XY(X_star, u_star, ax, vmin, vmax):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)

    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')

    im = ax.pcolor(X, Y, U_star, cmap='jet', vmin=vmin, vmax=vmax)
    return im


def plot_solution_XZ(X_star, u_star, ax, vmin, vmax):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    z = np.linspace(lb[1], ub[1], nn)
    X, Z = np.meshgrid(x, z)

    U_star = griddata(X_star, u_star.flatten(), (X, Z), method='cubic')

    im = ax.pcolor(X, Z, U_star, cmap='jet', vmin=vmin, vmax=vmax)
    return im


def plot_solution_YZ(X_star, u_star, ax, vmin, vmax):
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    y = np.linspace(lb[0], ub[0], nn)
    z = np.linspace(lb[1], ub[1], nn)
    Y, Z = np.meshgrid(y, z)

    U_star = griddata(X_star, u_star.flatten(), (Y, Z), method='cubic')

    im = ax.pcolor(Y, Z, U_star, cmap='jet', vmin=vmin, vmax=vmax)
    return im


def predict_XZ(data, model, t, vmin, vmax, interval):
    result = []
    vaules = np.arange(vmin, vmax + 1, interval)

    for y in vaules:
        plane_data = data[(data[:, 1] == y) & (data[:, 3] == t)][:, [0, 2, 4, 5, 6]]

        x_star = plane_data[:, 0].reshape(-1, 1)
        z_star = plane_data[:, 1].reshape(-1, 1)
        u_star = plane_data[:, 2].reshape(-1, 1)
        v_star = plane_data[:, 3].reshape(-1, 1)
        w_star = plane_data[:, 4].reshape(-1, 1)

        Nx = x_star.shape[0]

        speed_star = calc_v_magnitude(u_star, v_star, w_star)

        t_fixed = np.array([t]).reshape(-1, 1)
        y_fixed = np.array([y]).reshape(-1, 1)
        y_star = np.tile(y_fixed, (Nx, 1))  # Nx * Nz x 1
        t_star = np.tile(t_fixed, (Nx, 1))  # Nx * Nz x 1

        # Prediction
        u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
        speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)

        err = np.sqrt(np.mean((speed_star - speed_pred) ** 2))

        result.append(err)

    plt.figure()
    plt.plot(vaules, result, marker='o', linestyle='-')
    plt.xlabel('y'), plt.ylabel('err'), plt.title('X-Z Err')
    plt.grid(True), plt.savefig('X-Z.png'), plt.show()


def predict_XY(data, model, t, vmin, vmax, interval):
    result = []
    vaules = np.arange(vmin, vmax + 1, interval)

    for z in vaules:
        plane_data = data[(data[:, 2] == z) & (data[:, 3] == t)][:, [0, 1, 4, 5, 6]]

        x_star = plane_data[:, 0].reshape(-1, 1)
        y_star = plane_data[:, 1].reshape(-1, 1)
        u_star = plane_data[:, 2].reshape(-1, 1)
        v_star = plane_data[:, 3].reshape(-1, 1)
        w_star = plane_data[:, 4].reshape(-1, 1)

        Nx = x_star.shape[0]

        speed_star = calc_v_magnitude(u_star, v_star, w_star)

        t_fixed = np.array([t]).reshape(-1, 1)
        z_fixed = np.array([z]).reshape(-1, 1)
        z_star = np.tile(z_fixed, (Nx, 1))
        t_star = np.tile(t_fixed, (Nx, 1))

        # Prediction
        u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
        speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)

        err = np.sqrt(np.mean((speed_star - speed_pred) ** 2))

        result.append(err)

    plt.figure()
    plt.plot(vaules, result, marker='o', linestyle='-')
    plt.xlabel('z'), plt.ylabel('err'), plt.title('X-Y Err')
    plt.grid(True), plt.savefig('X-Y.png'), plt.show()


def predict_YZ(data, model, t, vmin, vmax, interval):
    result = []
    vaules = np.arange(vmin, vmax + 1, interval)

    for x in vaules:
        plane_data = data[(data[:, 0] == x) & (data[:, 3] == t)][:, [1, 2, 4, 5, 6]]

        y_star = plane_data[:, 0].reshape(-1, 1)
        z_star = plane_data[:, 1].reshape(-1, 1)
        u_star = plane_data[:, 2].reshape(-1, 1)
        v_star = plane_data[:, 3].reshape(-1, 1)
        w_star = plane_data[:, 4].reshape(-1, 1)

        Nx = y_star.shape[0]

        speed_star = calc_v_magnitude(u_star, v_star, w_star)

        t_fixed = np.array([t]).reshape(-1, 1)
        x_fixed = np.array([x]).reshape(-1, 1)
        x_star = np.tile(x_fixed, (Nx, 1))
        t_star = np.tile(t_fixed, (Nx, 1))

        # Prediction
        u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
        speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)

        err = np.sqrt(np.mean((speed_star - speed_pred) ** 2))

        result.append(err)

    plt.figure()
    plt.plot(vaules, result, marker='o', linestyle='-')
    plt.xlabel('x'), plt.ylabel('err'), plt.title('Y-Z Err')
    plt.grid(True), plt.savefig('Y-Z.png'), plt.show()


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def calc_v_np(u, v, w, phi, theda):
    theda_rad = theda * np.pi / 180
    phi_rad = phi * np.pi / 180
    v_r = (u * np.cos(theda_rad) * np.cos(phi_rad) +
           v * np.sin(theda_rad) * np.cos(phi_rad) +
           w * np.sin(phi_rad))

    return v_r


def predict_filed_XZ(data, model, rand_t, rand_y):
    plane_data = data[(data[:, 1] == rand_y) & (data[:, 3] == rand_t)][:, [0, 2, 4, 5, 6]]

    x_star = plane_data[:, 0].reshape(-1, 1)
    z_star = plane_data[:, 1].reshape(-1, 1)
    u_star = plane_data[:, 2].reshape(-1, 1)
    v_star = plane_data[:, 3].reshape(-1, 1)
    w_star = plane_data[:, 4].reshape(-1, 1)

    Nx = x_star.shape[0]

    speed_star = calc_v_magnitude(u_star, v_star, w_star)

    t_fixed = np.array([rand_t]).reshape(-1, 1)
    y_fixed = np.array([rand_y]).reshape(-1, 1)
    y_star = np.tile(y_fixed, (Nx, 1))  # Nx * Nz x 1
    t_star = np.tile(t_fixed, (Nx, 1))  # Nx * Nz x 1

    # Prediction
    u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
    speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)
    # Re_value = model.Re.item()  # Assuming re is a scalar tensor

    # error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    # error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    # error_w = np.linalg.norm(w_star-w_pred,2)/np.linalg.norm(w_star,2)
    error_speed = np.linalg.norm(speed_star - speed_pred, 2) / np.linalg.norm(speed_star, 2)
    # print('Error u: %e' % (error_u))
    # print('Error v: %e' % (error_v))
    # print('Error w: %e' % (error_w))
    print('Error speed: %e' % (error_speed))

    # Predict for plotting
    xz_location = plane_data[:, :2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    vmin = min(speed_pred.min(), speed_star.min())
    vmax = max(speed_pred.max(), speed_star.max())

    im1 = plot_solution_XZ(xz_location, speed_pred, ax1, vmin, vmax)
    ax1.set_title('Predicted Speed')
    im2 = plot_solution_XZ(xz_location, speed_star, ax2, vmin, vmax)
    ax2.set_title('True Speed')

    fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical', label='Speed')
    plt.savefig('X-Z_filed.png')
    # plt.tight_layout()
    plt.show()

    plot_solution_XZ_Single(xz_location, np.abs(speed_star - speed_pred), 3)


def predict_filed_YZ(data, model, rand_t, rand_x):
    plane_data = data[(data[:, 0] == rand_x) & (data[:, 3] == rand_t)][:, [1, 2, 4, 5, 6]]

    y_star = plane_data[:, 0].reshape(-1, 1)
    z_star = plane_data[:, 1].reshape(-1, 1)
    u_star = plane_data[:, 2].reshape(-1, 1)
    v_star = plane_data[:, 3].reshape(-1, 1)
    w_star = plane_data[:, 4].reshape(-1, 1)

    Nx = y_star.shape[0]

    speed_star = calc_v_magnitude(u_star, v_star, w_star)

    t_fixed = np.array([rand_t]).reshape(-1, 1)
    x_fixed = np.array([rand_x]).reshape(-1, 1)
    x_star = np.tile(x_fixed, (Nx, 1))  # Nx * Nz x 1
    t_star = np.tile(t_fixed, (Nx, 1))  # Nx * Nz x 1

    # Prediction
    u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
    speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)
    # Re_value = model.Re.item()  # Assuming re is a scalar tensor

    # error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    # error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    # error_w = np.linalg.norm(w_star-w_pred,2)/np.linalg.norm(w_star,2)
    error_speed = np.linalg.norm(speed_star - speed_pred, 2) / np.linalg.norm(speed_star, 2)
    # print('Error u: %e' % (error_u))
    # print('Error v: %e' % (error_v))
    # print('Error w: %e' % (error_w))
    print('Error speed: %e' % (error_speed))

    # Predict for plotting
    yz_location = plane_data[:, :2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    vmin = min(speed_pred.min(), speed_star.min())
    vmax = max(speed_pred.max(), speed_star.max())

    im1 = plot_solution_YZ(yz_location, speed_pred, ax1, vmin, vmax)
    ax1.set_title('Predicted Speed')
    im2 = plot_solution_YZ(yz_location, speed_star, ax2, vmin, vmax)
    ax2.set_title('True Speed')

    fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical', label='Speed')
    plt.savefig('Y-Z_filed.png')
    # plt.tight_layout()
    plt.show()

    plot_solution_YZ_Single(yz_location, np.abs(speed_star - speed_pred), 3)


def predict_filed_XY(data, model, rand_t, rand_z):
    plane_data = data[(data[:, 2] == rand_z) & (data[:, 3] == rand_t)][:, [0, 1, 4, 5, 6]]

    x_star = plane_data[:, 0].reshape(-1, 1)
    y_star = plane_data[:, 1].reshape(-1, 1)
    u_star = plane_data[:, 2].reshape(-1, 1)
    v_star = plane_data[:, 3].reshape(-1, 1)
    w_star = plane_data[:, 4].reshape(-1, 1)

    Nx = x_star.shape[0]

    speed_star = calc_v_magnitude(u_star, v_star, w_star)

    t_fixed = np.array([rand_t]).reshape(-1, 1)
    z_fixed = np.array([rand_z]).reshape(-1, 1)
    z_star = np.tile(z_fixed, (Nx, 1))  # Nx * Nz x 1
    t_star = np.tile(t_fixed, (Nx, 1))  # Nx * Nz x 1

    # Prediction
    u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
    speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)
    # Re_value = model.Re.item()  # Assuming re is a scalar tensor

    # error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    # error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    # error_w = np.linalg.norm(w_star-w_pred,2)/np.linalg.norm(w_star,2)
    error_speed = np.linalg.norm(speed_star - speed_pred, 2) / np.linalg.norm(speed_star, 2)
    # print('Error u: %e' % (error_u))
    # print('Error v: %e' % (error_v))
    # print('Error w: %e' % (error_w))
    print('Error speed: %e' % (error_speed))

    # Predict for plotting
    xy_location = plane_data[:, :2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    vmin = min(speed_pred.min(), speed_star.min())
    vmax = max(speed_pred.max(), speed_star.max())

    im1 = plot_solution_XY(xy_location, speed_pred, ax1, vmin, vmax)
    ax1.set_title('Predicted Speed')
    im2 = plot_solution_XY(xy_location, speed_star, ax2, vmin, vmax)
    ax2.set_title('True Speed')

    fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical', label='Speed')
    plt.savefig('X-Y_filed.png')
    # plt.tight_layout()
    plt.show()

    plot_solution_XY_Single(xy_location, np.abs(speed_star - speed_pred), 3)


def extract_and_round_data_all(df):
    x = df.iloc[:, 0].values.reshape(-1, 1).round(1)
    y = df.iloc[:, 1].values.reshape(-1, 1).round(1)
    z = df.iloc[:, 2].values.reshape(-1, 1).round(1)
    t = df.iloc[:, 3].values.reshape(-1, 1)
    u = df.iloc[:, 4].values.reshape(-1, 1)
    v = df.iloc[:, 5].values.reshape(-1, 1)
    w = df.iloc[:, 6].values.reshape(-1, 1)
    phi = df.iloc[:, 7].values.reshape(-1, 1)
    theda = df.iloc[:, 8].values.reshape(-1, 1)
    return x, y, z, t, u, v, w, phi, theda


def extract_and_round_data(x_input, y_input, z_input, t_input, u_input, v_input, w_input, phi_input, theda_input, data_type):
    if data_type == 'LiDar':
        index = phi_input != -888
    elif data_type == 'RandomPoints':
        index = phi_input == -888
    x = x_input[index]
    y = y_input[index]
    z = z_input[index]
    t = t_input[index]
    u = u_input[index]
    v = v_input[index]
    w = w_input[index]
    phi = phi_input[index]
    theda = theda_input[index]
    return x, y, z, t, u, v, w, phi, theda


def findLastPth(path):
    try:
        pth_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pth')]
        lastPth = max(pth_files, key=os.path.getctime)
    except ValueError:
        lastPth = None
    return lastPth


def combine_tensors_random1_lidar2(tensor_random, tensor_lidar):
    reshaped_tensor_lidar = tensor_random.unsqueeze(1).view(20000, -1)
    return torch.cat((reshaped_tensor_lidar, tensor_lidar.unsqueeze(1)), dim=1)
    # return torch.cat((tensor_lidar.unsqueeze(1), reshaped_tensor_lidar), dim=1)

# def combine_tensors_random1_lidar2(arr_random, arr_lidar):
#     reshaped_arr_lidar = np.expand_dims(arr_random, axis=1).reshape(20000, -1)
#     return np.concatenate((reshaped_arr_lidar, np.expand_dims(arr_lidar, axis=1)), axis=1)
