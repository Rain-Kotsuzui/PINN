# ut+uux=nu*uxx
# nu=0.01
# x∈[0,2],t∈[0,3]
# u(0,t)=0,u(2,t)=0,u(x,0)=-sin(pi*(x-1))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def sample_points(device, n=5000, x_=2.0, t_=3.0):
    x = 1.0+0.5*torch.randn(n, 1, device=device)  # x∈[0,2]
    x = torch.clamp(x, 0.0, x_)
    t = torch.rand(n, 1, device=device)*t_  # t∈[0,3]
    return x.requires_grad_(True), t.requires_grad_(True)


def left(device, n=100, x=2.0, t=3.0):
    t = torch.rand(n, 1, device=device)*t  # t∈[0,3]
    x = torch.zeros(n, 1, device=device)  # x=0
    cond = torch.zeros(n, 1, device=device)
    return x.requires_grad_(True), t.requires_grad_(True), cond.requires_grad_(True)


def right(device, n=100, x=2.0, t=3.0):
    t = torch.rand(n, 1, device=device)*t  # t∈[0,3]
    x = torch.ones(n, 1, device=device)*x  # x=2
    cond = torch.zeros(n, 1, device=device)
    return x.requires_grad_(True), t.requires_grad_(True), cond.requires_grad_(True)


def initial(device, n=100, x=2.0, t=3.0):
    x = torch.rand(n, 1, device=device)*x  # x∈[0,2]
    t = torch.zeros(n, 1, device=device)  # t=0
    cond = -torch.sin(torch.pi*(x-1.0))
    return x.requires_grad_(True), t.requires_grad_(True), cond.requires_grad_(True)


# 傅里叶特征映射层
class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dims, mapping_size=256, scale=10.0):
        super().__init__()
        self.input_dims = input_dims
        self.mapping_size = mapping_size
        B = torch.randn((input_dims, mapping_size)) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = x @ self.B
        return torch.cat([torch.sin(2 * np.pi * x_proj), torch.cos(2 * np.pi * x_proj)], dim=-1)

class FourierNet(nn.Module):
    def __init__(self, input_dims=2, mapping_size=256, scale=10.0):
        super().__init__()
        self.fourier_mapper = FourierFeatureMapping(input_dims, mapping_size, scale)
        mlp_input_dims = 2 * mapping_size

        self.net = nn.Sequential(
            nn.Linear(mlp_input_dims, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        fourier_features = self.fourier_mapper(x)
        return self.net(fourier_features)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
             nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 1)
        )

    def forward(self, x):
        return self.net(x)


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order-1)


loss = nn.MSELoss()


def l_sample_we(device, nu, u, k1=0.2):
    x, t = sample_points(device)
    u_val = u(torch.cat([x, t], dim=1))

    u_t = gradients(u_val, t, 1)
    u_x = gradients(u_val, x, 1)
    
    u_xx = 0.0
    if nu >= 0.000001:
        u_xx = gradients(u_val, x, 2)
    #  1 / (k1 * (|u_x| - u_x) + 1)
    lambda_1 = 1.0 / (k1 * (torch.abs(u_x) - u_x) + 1.0)

    pde_residual = u_t + u_val * u_x - nu * u_xx
    weighted_pde_residual = lambda_1 * pde_residual
    return loss(weighted_pde_residual, torch.zeros_like(weighted_pde_residual))


def l_rh(device, u, n=100, t_=3.0):
    t = torch.rand(n, 1, device=device)*t_  # t∈[0,3]
    x = torch.ones(n, 1, device=device)  # 不连续位置 x = 1

    u_val = u(torch.cat([x, t], dim=1))

    # u(1, t)= u(1, 0) = -sin(pi*(1-1)) = 0
    rh_cond = torch.zeros_like(u_val)

    return loss(u_val, rh_cond)


def l_sample(device, nu: float, u):
    x, t = sample_points(device)
    uxy = u(torch.cat([x, t], dim=1))  # uxy=u(x,t)
    u_xx=0.0
    if nu >= 0.000001:
        u_xx = gradients(uxy, x, 2)
    return loss(gradients(uxy, t, 1) + uxy*gradients(uxy, x, 1) - nu*u_xx, torch.zeros_like(uxy))


def l_boundary(device, u):
    x_l, t_l, cond_l = left(device)
    x_r, t_r, cond_r = right(device)
    u_l = u(torch.cat([x_l, t_l], dim=1))
    u_r = u(torch.cat([x_r, t_r], dim=1))
    return loss(u_l, cond_l) + loss(u_r, cond_r)


def l_initial(device, u):
    x, t, cond = initial(device)
    uxy = u(torch.cat([x, t], dim=1))  # uxy=u(x,t)
    return loss(uxy, cond)


def sol_we(nu: float, x: float, t: float, grid: int, training_times: int) -> None:
    device = torch.device("cuda")
    print(f"Using device: {device}")
    #u = FourierNet(input_dims=2, mapping_size=256, scale=0.40).to(device)
    u = MLP().to(device)
    opt = optim.Adam(u.parameters(), lr=1e-3)

    # scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    omega_pde = 1.0
    omega_ibc = 15.0
    omega_rh = 5.0
    import time
    start_time = time.time()

    for epoch in range(training_times):
        opt.zero_grad()
        l = omega_pde*l_sample_we(device, nu, u) +\
            omega_ibc*(l_boundary(device, u)+l_initial(device, u)) +\
            omega_rh * l_rh(device, u)

        l.backward()
        opt.step()
        # if (epoch+1) % 500 == 0:
        #    scheduler.step()
        if epoch % 100 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            print(
                f"epoch:{epoch},loss:{l.item()},remain:{(training_times-epoch)*elapsed_time/(epoch+1):.2f}s")

    print("Training finished!")
    with torch.no_grad():
        xc = torch.linspace(0, x, grid, device=device)
        tc = torch.linspace(0, t, grid, device=device)
        xx, tt = torch.meshgrid(xc, tc, indexing='ij')

        xt_grid = torch.cat([xx.reshape(-1, 1), tt.reshape(-1, 1)], dim=1)
        u_pred = u(xt_grid)

        u_pred_cpu = u_pred.reshape(grid, grid).cpu().numpy()
        x_cpu = xx.cpu().numpy()
        t_cpu = tt.cpu().numpy()
        draw(u_pred_cpu, x_cpu, t_cpu)


def sol(nu: float, x: float, t: float, grid: int, training_times: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # u = FourierNet(input_dims=2, mapping_size=256, scale=0.40).to(device)
    u = MLP().to(device)
    opt = optim.Adam(u.parameters(), lr=1e-3)

    # scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    omega_pde = 1.0
    omega_ibc = 1.0

    import time
    start_time = time.time()

    for epoch in range(training_times):
        opt.zero_grad()
        l = omega_pde*l_sample_we(device, nu, u) +\
            omega_ibc*(l_boundary(device, u)+l_initial(device, u))

        l.backward()
        opt.step()
        # if (epoch+1) % 500 == 0:
        #    scheduler.step()
        if epoch % 100 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            print(
                f"epoch:{epoch},loss:{l.item()},remain:{(training_times-epoch)*elapsed_time/(epoch+1):.2f}s")

    print("Training finished!")
    with torch.no_grad():
        xc = torch.linspace(0, x, grid, device=device)
        tc = torch.linspace(0, t, grid, device=device)
        xx, tt = torch.meshgrid(xc, tc, indexing='ij')

        xt_grid = torch.cat([xx.reshape(-1, 1), tt.reshape(-1, 1)], dim=1)
        u_pred = u(xt_grid)

        u_pred_cpu = u_pred.reshape(grid, grid).cpu().numpy()
        x_cpu = xx.cpu().numpy()
        t_cpu = tt.cpu().numpy()
        draw(u_pred_cpu, x_cpu, t_cpu)


def draw(u_pred, x, t):

    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(x, t, u_pred, cmap=plt.cm.cividis)
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)
    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.show()

    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    PINN_path = os.path.join(current_dir, 'burgers_PINN.npy')
    np.save(PINN_path, {'x': x, 't': t,
            'u': u_pred})


if __name__ == '__main__':
    sol(0.01, 2.0, 3.0, 100, 3000)
