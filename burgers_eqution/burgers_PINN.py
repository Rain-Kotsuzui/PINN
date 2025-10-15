# ut+uux=nu*uxx
# nu=0.01
# x∈[0,2],t∈[0,3]
# u(0,t)=0,u(2,t)=0,u(x,0)=-sin(pi*(x-1))

import torch
import torch.nn as nn
import torch.optim as optim


def sample_points(device, n=5000, x=2.0, t=3.0):
    x = torch.rand(n, 1, device=device)*x  # x∈[0,2]
    t = torch.rand(n, 1, device=device)*3.0  # t∈[0,3]
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
    cond = -torch.sin(torch.pi*(x-1))
    return x.requires_grad_(True), t.requires_grad_(True), cond.requires_grad_(True)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 75),
            nn.Tanh(),
            nn.Linear(75, 100),
            nn.Tanh(),
            nn.Linear(100, 70),
            nn.Tanh(),
            nn.Linear(70, 1)
        )

    def forward(self, x):
        return self.net(x)


def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order-1)


loss = nn.MSELoss()


def l_sample(device, nu: float, u):
    x, t = sample_points(device)
    uxy = u(torch.cat([x, t], dim=1))  # uxy=u(x,t)
    return loss(gradients(uxy, t, 1) + uxy*gradients(uxy, x, 1) - nu*gradients(uxy, x, 2), torch.zeros_like(uxy))


def l_left(device, u):
    x, t, cond = left(device)
    uxy = u(torch.cat([x, t], dim=1))  # uxy=u(x,t)
    return loss(uxy, cond)


def l_right(device, u):
    x, t, cond = right(device)
    uxy = u(torch.cat([x, t], dim=1))  # uxy=u(x,t)
    return loss(uxy, cond)


def l_initial(device, u):
    x, t, cond = initial(device)
    uxy = u(torch.cat([x, t], dim=1))  # uxy=u(x,t)
    return loss(uxy, cond)


def sol(nu: float, x: float, t: float, grid:int,training_times:int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    u = MLP().to(device)  # 将模型移动到GPU
    opt = optim.Adam(u.parameters(), lr=0.001)

    omega_data = 0.5
    omega_pde = 1-omega_data

    import time
    start_time = time.time()

    for epoch in range(training_times):
        opt.zero_grad()
        l = omega_pde*l_sample(device, nu, u) +\
            omega_data*(l_left(device, u) +
                        l_right(device, u)+l_initial(device, u))

        l.backward()
        opt.step()
        if epoch % 100 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            print(f"epoch:{epoch},loss:{l.item()},remain:{(training_times-epoch)*elapsed_time/(epoch+1):.2f}s")

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
    np.save('burgers_PINN', {'x': x, 't': t,
            'u': u_pred})
