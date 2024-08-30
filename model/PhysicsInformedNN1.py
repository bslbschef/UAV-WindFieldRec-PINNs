import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from util.tools import extract_and_round_data


class PhysicsInformedNN(nn.Module):
    def __init__(self, x, y, z, t, u, v, w, phi, theda, layers, device):
        super(PhysicsInformedNN, self).__init__()

        self.phi = phi
        self.theda = theda

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
        self.register_parameter("Reciprocal_Re", nn.Parameter(torch.tensor([0.0001], device=device)))

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
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = nn.Parameter(torch.zeros(1, layers[l + 1]))
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

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = torch.tanh(torch.add(torch.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = torch.add(torch.matmul(H, W), b)

        return Y

    def net_NS(self, x, y, z, t):
        Reciprocal_Re = self.Reciprocal_Re

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

        f_u = u_t + u * u_x + v * u_y + w * u_z + p_x - Reciprocal_Re * (u_xx + u_yy + u_zz)
        f_v = v_t + u * v_x + v * v_y + w * v_z + p_y - Reciprocal_Re * (v_xx + v_yy + v_zz)
        f_w = w_t + u * w_x + v * w_y + w * w_z + p_z - Reciprocal_Re * (w_xx + w_yy + w_zz)

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
        return torch.sqrt(u ** 2 + v ** 2 + w ** 2) / 8

    def forward(self, x, y, z, t, data_type):
        self.Reciprocal_Re.data.clamp_(min=1e-8)
        u_pred, v_pred, w_pred, p_pred, f_u_pred, f_v_pred, f_w_pred, f_e_pred = self.net_NS(x, y, z, t)

        if data_type != 'LiDAR':  # LiDAR measured speed (line of sight)
            speed_pred = self.calc_v(u_pred, v_pred, w_pred, self.phi, self.theda)
            speed = self.calc_v(self.u, self.v, self.w, self.phi, self.theda)
            data_loss = torch.mean((speed - speed_pred) ** 2)
            equation_loss = torch.mean(f_u_pred ** 2) + torch.mean(f_v_pred ** 2) + torch.mean(f_w_pred ** 2) + torch.mean(f_e_pred ** 2)
            loss = (data_loss + equation_loss) / 2
        elif data_type == 'RandomPoints':
            equation_loss = torch.mean(f_u_pred ** 2) + torch.mean(f_v_pred ** 2) + torch.mean(f_w_pred ** 2) + torch.mean(f_e_pred ** 2)
            loss = equation_loss
        return loss

    def train(self, nIter, save_path):
        def closure(x, y, z, t, data_type):
            self.optimizer.zero_grad()
            loss = self.forward(x, y, z, t, data_type)
            loss.backward()
            # print(f"Re: {self.Re.item()}, Re grad: {self.Re.grad}")
            # print(f"Re requires grad: {self.Re.requires_grad}")
            return loss

        writer = SummaryWriter(log_dir=os.path.join(save_path, 'tensorboard_logs'))
        global_step = 0
        start_time = time.time()
        total_loss = 0

        x_random_points, y_random_points, z_random_points, t_random_points, u_random_points, v_random_points, w_random_points, phi_random_points, theda_random_points = extract_and_round_data(
            self.x, self.y, self.z, self.u,
            self.v, self.w, self.phi, self.theda, 'random_points')
        x_lidar, y_lidar, z_lidar, t_lidar, u_lidar, v_lidar, w_lidar, phi_lidar, theda_lidar = extract_and_round_data(
            self.x, self.y, self.z, self.u, self.v, self.w, self.phi, self.theda, 'lidar')

        for it in range(nIter):
            self.optimizer.step(closure(x, y, z, t, data_type))

            if (it + 1) % 100 == 0:
                elapsed = time.time() - start_time
                loss_value = self.forward().item()
                Re_value = self.Reciprocal_Re.item()
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
                torch.save(self.state_dict(), os.path.join(save_path, f'PINN_{it + 1}.pth'))
                print("Model weights saved at iteration: ", it + 1)
                total_loss = 0

                # torch.save({
                #     'epoch': it+1,
                #     'model_state_dict': self.state_dict(),
                #     'optimizer_state_dict': self.optimizer_lbfgs.state_dict(),
                #     'loss': self.forward().item(),
                #     'learning_rate': self.optimizer_lbfgs.param_groups[0]['lr'],
                # }, os.path.join(save_path, f'PINN_{it + 1}.pth'))

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
