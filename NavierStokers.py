import time
import os
import scipy.io
import torch
import scipy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import h5py
from torch.utils.tensorboard import SummaryWriter


class PhysicsInformedNN(nn.Module):
    def __init__(self, x, y, z, t, u, v, w, layers, device):
        super(PhysicsInformedNN, self).__init__()

        X = np.concatenate([x, y, z, t], axis=1)  # N x 4
        self.lb = torch.tensor(X.min(0), dtype=torch.float32).to(device)
        self.ub = torch.tensor(X.max(0), dtype=torch.float32).to(device)
        self.X = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)

        self.x = self.X[:, 0:1]
        self.y = self.X[:, 1:2]
        self.z = self.X[:, 2:3]
        self.t = self.X[:, 3:4]

        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        self.v = torch.tensor(v, dtype=torch.float32).to(device)
        self.w = torch.tensor(w, dtype=torch.float32).to(device)
    
        self.layers = layers

        self.weights, self.biases = self.initialize_NN(layers)
        
        # self.Re = nn.Parameter(torch.tensor([0.0], dtype=torch.float32, requires_grad=True)).to(device)
        self.register_parameter("Re", nn.Parameter(torch.tensor([100.0], device=device)))
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.optimizer_lbfgs = optim.LBFGS(self.parameters(), 
                                           max_iter=50000, 
                                           max_eval=50000, 
                                           tolerance_grad=1e-9, 
                                           tolerance_change=1e-7,
                                           history_size=50,
                                           line_search_fn='strong_wolfe')
        
        self.weights.to(device)
        self.biases.to(device)
        self.device = device
        
    def initialize_NN(self, layers):
        weights = nn.ParameterList()
        biases = nn.ParameterList()
        num_layers = len(layers)
        
        for l in range(num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = nn.Parameter(torch.zeros(1, layers[l+1]))
            weights.append(W)
            biases.append(b)
        
        return weights, biases
    
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
        return nn.Parameter(torch.randn(size) * xavier_stddev, requires_grad=True)

    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H  = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = torch.tanh(torch.add(torch.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = torch.add(torch.matmul(H, W), b)

        return Y
    
    def net_NS(self, x, y, z, t):
        Re = self.Re
        
        outputs = self.neural_net(torch.cat([x, y, z, t], dim=1), self.weights, self.biases)
        u, v, w, p = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:, 3:4]
        
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]
        
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
        v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]

        w_t = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
        w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
        w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        
        f_u = u_t + u * u_x + v * u_y + w * u_z + p_x - (1 / Re) * (u_xx + u_yy + u_zz)
        f_v = v_t + u * v_x + v * v_y + w * v_z + p_y - (1 / Re) * (v_xx + v_yy + v_zz)
        f_w = w_t + u * w_x + v * w_y + w * w_z + p_z - (1 / Re) * (w_xx + w_yy + w_zz)

        f_e = u_x + v_y + w_z
        
        return u, v, w, p, f_u, f_v, f_w, f_e
    
    def calc_v(self, u, v, w, phi, theda):
        theda_rad = theda * torch.pi / 180
        phi_rad = phi * torch.pi / 180
        v_r = (u * torch.cos(theda_rad) * torch.cos(phi_rad) + 
                v * torch.sin(theda_rad) * torch.cos(phi_rad) +
                w * torch.sin(phi_rad))

        return v_r
    
    def calc_v_magnitude(self, u, v, w):
        U = 8
        return torch.sqrt(u**2 + v**2 + w**2) / 8
    
    def forward(self):
        self.Re.data.clamp_(min=1e-8)
        u_pred, v_pred, w_pred, p_pred, f_u_pred, f_v_pred, f_w_pred, f_e_pred = self.net_NS(self.x, self.y, self.z, self.t)

        # speed_pred = self.calc_v(u_pred, v_pred, w_pred, self.phi, self.theda)
        # speed = self.calc_v(self.u, self.v, self.w, self.phi, self.theda)
        # data_loss = torch.mean((self.u - u_pred)**2) + torch.mean((self.v - v_pred)**2) + torch.mean((self.w - w_pred)**2)
        speed_pred = self.calc_v_magnitude(u_pred, v_pred, w_pred)
        speed = self.calc_v_magnitude(self.u, self.v, self.w)
        data_loss = torch.mean((speed - speed_pred) ** 2)
        equation_loss = torch.mean(f_u_pred**2) + torch.mean(f_v_pred**2) +torch.mean(f_w_pred**2) + torch.mean(f_e_pred**2)
        loss = (data_loss + equation_loss) / 2
        return loss

    def train(self, nIter, save_path):
        def closure():
            self.optimizer.zero_grad()
            loss = self.forward()
            loss.backward()
            # print(f"Re: {self.Re.item()}, Re grad: {self.Re.grad}")
            # print(f"Re requires grad: {self.Re.requires_grad}")
            return loss

        writer = SummaryWriter(log_dir=os.path.join(save_path, 'tensorboard_logs'))
        global_step = 0
        start_time = time.time()
        total_loss = 0
        for it in range(nIter):
            self.optimizer.step(closure)

            if (it + 1) % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.forward().item()
                Re_value = self.Re.item()
                total_loss += loss_value
                print('It: %d, Loss: %.3f, Re: %.5f, Time: %.2f' % 
                      (it + 1, loss_value, Re_value, elapsed))
                writer.add_scalar('Loss/iter', loss_value, global_step)
                writer.add_scalar('Params/Re', Re_value, global_step)
                global_step += 100
                start_time = time.time()
            
            if (it + 1) % 10000 == 0:
                avg_loss = total_loss / 100
                writer.add_scalar('Loss/epoch', avg_loss, global_step / 10000)
                torch.save(self.state_dict(), os.path.join(save_path, f'PINN_{it+1}.pth'))
                print("Model weights saved at iteration: ", it + 1)
                total_loss = 0
        
        print("Training is Finish!!!")
        writer.close()

    # def predict(self, x_star, y_star, z_star, t_star, phi_star, theda_star):
    #     x_star_tensor = torch.tensor(x_star, dtype=torch.float32, requires_grad=True, device=self.device)
    #     y_star_tensor = torch.tensor(y_star, dtype=torch.float32, requires_grad=True, device=self.device)
    #     z_star_tensor = torch.tensor(z_star, dtype=torch.float32, requires_grad=True, device=self.device)
    #     t_star_tensor = torch.tensor(t_star, dtype=torch.float32, requires_grad=True, device=self.device)

    #     phi_star = torch.tensor(phi_star, dtype=torch.float32, device=self.device)
    #     theda_star = torch.tensor(theda_star, dtype=torch.float32, device=self.device)

    #     u_star, v_star, w_star, _, _, _, _, _ = self.net_NS(x_star_tensor, y_star_tensor, z_star_tensor, t_star_tensor)
    #     speed_star = self.calc_v(u_star, v_star, w_star, phi_star, theda_star)

    #     u_star = u_star.cpu().detach().numpy()
    #     v_star = v_star.cpu().detach().numpy()
    #     w_star = w_star.cpu().detach().numpy()
    #     speed_star = speed_star.cpu().detach().numpy()

    #     return u_star, v_star, w_star, speed_star
    def predict(self, x_star, y_star, z_star, t_star):
        x_star_tensor = torch.tensor(x_star, dtype=torch.float32, requires_grad=True, device=self.device)
        y_star_tensor = torch.tensor(y_star, dtype=torch.float32, requires_grad=True, device=self.device)
        z_star_tensor = torch.tensor(z_star, dtype=torch.float32, requires_grad=True, device=self.device)
        t_star_tensor = torch.tensor(t_star, dtype=torch.float32, requires_grad=True, device=self.device)


        u_star, v_star, w_star, _, _, _, _, _ = self.net_NS(x_star_tensor, y_star_tensor, z_star_tensor, t_star_tensor)

        u_star = u_star.cpu().detach().numpy()
        v_star = v_star.cpu().detach().numpy()
        w_star = w_star.cpu().detach().numpy()

        return u_star, v_star, w_star
    

def calc_v_magnitude(u, v, w):
    return np.sqrt(u**2 + v**2 + w**2)


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
        y_star = np.tile(y_fixed, (Nx, 1))   # Nx * Nz x 1
        t_star = np.tile(t_fixed, (Nx, 1))   # Nx * Nz x 1

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
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
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
    y_star = np.tile(y_fixed, (Nx, 1))   # Nx * Nz x 1
    t_star = np.tile(t_fixed, (Nx, 1))   # Nx * Nz x 1

    # Prediction
    u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
    speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)
    # Re_value = model.Re.item()  # Assuming re is a scalar tensor
    
    # error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    # error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    # error_w = np.linalg.norm(w_star-w_pred,2)/np.linalg.norm(w_star,2)
    error_speed = np.linalg.norm(speed_star-speed_pred,2)/np.linalg.norm(speed_star,2)
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
    x_star = np.tile(x_fixed, (Nx, 1))   # Nx * Nz x 1
    t_star = np.tile(t_fixed, (Nx, 1))   # Nx * Nz x 1

    # Prediction
    u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
    speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)
    # Re_value = model.Re.item()  # Assuming re is a scalar tensor
    
    # error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    # error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    # error_w = np.linalg.norm(w_star-w_pred,2)/np.linalg.norm(w_star,2)
    error_speed = np.linalg.norm(speed_star-speed_pred,2)/np.linalg.norm(speed_star,2)
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
    z_star = np.tile(z_fixed, (Nx, 1))   # Nx * Nz x 1
    t_star = np.tile(t_fixed, (Nx, 1))   # Nx * Nz x 1

    # Prediction
    u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
    speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)
    # Re_value = model.Re.item()  # Assuming re is a scalar tensor
    
    # error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    # error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    # error_w = np.linalg.norm(w_star-w_pred,2)/np.linalg.norm(w_star,2)
    error_speed = np.linalg.norm(speed_star-speed_pred,2)/np.linalg.norm(speed_star,2)
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


if __name__ == "__main__":
    N_train = 97000 # sparsity: 0.14%

    layers = [4, 100, 100, 100, 100, 4]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    with h5py.File('data_uvw.mat', 'r') as file:
    # 注意：在 HDF5 格式中，MATLAB 数组是转置的
        data = file['data_uvw'][:].T # NT x 7
    
    
    x = data[:, 0].reshape(-1, 1)  # NT x 1
    y = data[:, 1].reshape(-1, 1)
    z = data[:, 2].reshape(-1, 1)
    t = data[:, 3].reshape(-1, 1)
    u = data[:, 4].reshape(-1, 1)
    v = data[:, 5].reshape(-1, 1)
    w = data[:, 6].reshape(-1, 1)


    # print("\nBasic statistics:")
    # print("X range:", np.min(x), "-", np.max(x))
    # print("Y range:", np.min(y), "-", np.max(y))
    # print("Z range:", np.min(z), "-", np.max(z))
    # print("T range:", np.min(t), "-", np.max(t))
    # print("V range:", np.min(v), "-", np.max(v))

    NT = x.shape[0]
    
    ######################################################################
    ######################## Noiseless Data ###############################
    ######################################################################
    # Training Data    
    idx = np.random.choice(NT, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    z_train = z[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]
    w_train = w[idx,:]

    ######################################################################
    ######################## LiDAR Data ###############################
    ######################################################################
    # train_data = scipy.io.loadmat('data_angle.mat')['data']
    # x_train = train_data[:, 0].reshape(-1, 1)  # NT x 1
    # y_train = train_data[:, 1].reshape(-1, 1)
    # z_train = train_data[:, 2].reshape(-1, 1)
    # t_train = train_data[:, 3].reshape(-1, 1)
    # u_train = train_data[:, 4].reshape(-1, 1)
    # v_train = train_data[:, 5].reshape(-1, 1)
    # w_train = train_data[:, 6].reshape(-1, 1)
    # phi = train_data[:, 7].reshape(-1, 1)
    # theda = train_data[:, 8].reshape(-1, 1)

    model = PhysicsInformedNN(x_train, y_train, z_train, t_train, u_train, v_train, w_train, layers, device)

    mode = 'Test'

    if mode == 'Training':
        # Training
        # ----------------------------------------------------------------------
        # print("Model parameters:")
        # for name, param in model.named_parameters():
        #     print(f"Parameter: {name}, Size: {param.size()}, Requires grad: {param.requires_grad}")

        # print("\nOptimizer parameters:")
        # for group in model.optimizer.param_groups:
        #     for p in group['params']:
        #         print(f"Parameter size: {p.size()}, requires_grad: {p.requires_grad}")

        # # 测试前向传播和反向传播
        # test_loss = model.forward()
        # test_loss.backward()
        # print(f"\nInitial Re: {model.Re.item()}, Initial Re grad: {model.Re.grad}")

        # # 简单的 Re 更新测试
        # print("\nSimple Re update test:")
        # for i in range(5):
        #     loss = (model.Re - 1000) ** 2
        #     loss.backward()
        #     print(f"Iteration {i}, Re: {model.Re.item()}, Re grad: {model.Re.grad}")
        #     model.optimizer.step()
        #     model.optimizer.zero_grad()
        # -----------------------------------------------------------------------
        model.train(500000, "save/")
    elif mode == 'Test':
        
        model.load_state_dict(torch.load('save/PINN_500000.pth'))

        # Test Data
        rand_t = 98.5
        xmax, xmin = x.max(), x.min()
        ymax, ymin = y.max(), y.min()
        zmax, zmin = z.max(), z.min()
        x_interval, y_interval, z_interval = 10, 10, 7.5
        
        
        # predict_XY(data, model, rand_t, zmin, zmax, z_interval)
        # predict_XZ(data, model, rand_t, ymin, ymax, y_interval)
        # predict_YZ(data, model, rand_t, xmin, xmax, x_interval)

        rand_x, rand_y, rand_z = 200, 100, 172.5
        predict_filed_XY(data, model, rand_t, rand_z)
        predict_filed_YZ(data, model, rand_t, rand_x)
        predict_filed_XZ(data, model, rand_t, rand_y)


        rand_y = 190
        plane_data = data[(data[:, 1] == rand_y) & (data[:, 3] == rand_t)][:, [0, 2, 4, 5, 6]]
        # print("Shape of plane_data:", plane_data.shape)
        # print(plane_data[0, 1])
        
        x_star = plane_data[:, 0].reshape(-1, 1)
        z_star = plane_data[:, 1].reshape(-1, 1)
        u_star = plane_data[:, 2].reshape(-1, 1)
        v_star = plane_data[:, 3].reshape(-1, 1)
        w_star = plane_data[:, 4].reshape(-1, 1)

        Nx = x_star.shape[0]
        # plane_data_angle = train_data[train_data[:, 3] == rand_t][:, [7, 8]]
        # phi_star = plane_data_angle[0, 0].reshape(-1, 1)
        # theda_star = plane_data_angle[0, 1].reshape(-1, 1)
        # # print(phi_star)
        # # print(theda_star)
        # phi_star = np.tile(phi_star, (Nx, 1))
        # theda_star = np.tile(theda_star, (Nx, 1))

        speed_star = calc_v_magnitude(u_star, v_star, w_star)

        t_fixed = np.array([rand_t]).reshape(-1, 1)
        y_fixed = np.array([rand_y]).reshape(-1, 1)
        y_star = np.tile(y_fixed, (Nx, 1))   # Nx * Nz x 1
        t_star = np.tile(t_fixed, (Nx, 1))   # Nx * Nz x 1

        # Prediction
        u_pred, v_pred, w_pred = model.predict(x_star, y_star, z_star, t_star)
        speed_pred = calc_v_magnitude(u_pred, v_pred, w_pred)
        # Re_value = model.Re.item()  # Assuming re is a scalar tensor
        
        # error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
        # error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
        # error_w = np.linalg.norm(w_star-w_pred,2)/np.linalg.norm(w_star,2)
        error_speed = np.linalg.norm(speed_star-speed_pred,2)/np.linalg.norm(speed_star,2)
        # print('Error u: %e' % (error_u)) 
        # print('Error v: %e' % (error_v)) 
        # print('Error w: %e' % (error_w)) 
        print('Error speed: %e' % (error_speed)) 
        #     if error_speed < error_min:
        #         error_min = error_speed
        #         xz_min = rand_y

        # Predict for plotting
        xz_location = plane_data[:, :2]
        # plot_solution_XZ(xz_location, u_pred, 1)
        # plot_solution_XZ(xz_location, v_pred, 2)
        # plot_solution_XZ(xz_location, w_pred, 3)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        vmin = min(speed_pred.min(), speed_star.min())
        vmax = max(speed_pred.max(), speed_star.max())

        im1 = plot_solution_XZ(xz_location, speed_pred, ax1, vmin, vmax)
        ax1.set_title('Predicted Speed')
        im2 = plot_solution_XZ(xz_location, speed_star, ax2, vmin, vmax)
        ax2.set_title('True Speed')
        
        fig.colorbar(im2, ax=[ax1, ax2], orientation='vertical', label='Speed')

        # plt.tight_layout()
        plt.show()
        
        plot_solution_XZ_Single(xz_location, np.abs(speed_star - speed_pred), 4)
        