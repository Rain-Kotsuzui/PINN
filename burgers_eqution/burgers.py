# ut+uux=nu*uxx
# nu=0.01
# x∈[0,2],t∈[0,3]
# u(0,t)=0,u(2,t)=0,u(x,0)=-sin(pi*(x-1))
import threading
import torch
import scipy.io
import numpy as np
from torch import nn
import burgers_PINN
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
def matlab_accsol(nu: float, x: float, t: float, grid: int) -> None:
    print("Matlab exact solution")
    global eng
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd(current_dir) 
    #eng.burgers_no_mu(x, t, grid, nargout=0)
    eng.burgers(nu,x, t, grid, nargout=0)


def PINN(nu: float, x: float, t: float, grid: int,training_times:int) -> None:
    print("PINN")
    burgers_PINN.sol_we(nu, x, t, grid,training_times)


def compare():
    
    mat_path = os.path.join(current_dir, 'burgers_no_mu.mat')
    mat = scipy.io.loadmat(mat_path)
    u_mat = mat['u'].T
    PINN_path = os.path.join(current_dir, 'burgers_PINN.npy')
    data = np.load(PINN_path, allow_pickle=True).item()
    u_PINN = data.get('u')
    x_PINN = data.get('x')
    t_PINN = data.get('t')

    print(np.max(np.abs(u_PINN-u_mat)))
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    dis=np.abs(u_PINN-u_mat)
    surf = ax.plot_surface(x_PINN,t_PINN,dis, cmap=plt.cm.cividis)
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('t', labelpad=20)
    ax.set_zlabel('u', labelpad=20)
    fig1.suptitle(f'Max error:{np.max(np.abs(u_PINN-u_mat))}')
    fig1.colorbar(surf, shrink=0.5, aspect=8)

    
    fig2 = plt.figure(figsize=(5, 5))
    x_mat_vec = mat['x'].flatten()
    t_mat_vec = mat['t'].flatten()
    x_pinn_vec = data.get('x')[:, 0]
    t_pinn_vec = data.get('t')[0, :]
    target_t = 1.0
    
    # 找到最接近 t=1 的时间索引
    idx_mat = np.argmin(np.abs(t_mat_vec - target_t))
    idx_pinn = np.argmin(np.abs(t_pinn_vec - target_t))
    
    # 提取该时刻的解
    u_mat_t1 = u_mat[:,idx_mat]
    u_pinn_t1 = u_PINN[:, idx_pinn]
    
    # --- 绘图 ---
    plt.plot(x_mat_vec, u_mat_t1, 'k-', linewidth=2, label='Analytical Solution')
    plt.plot(x_pinn_vec, u_pinn_t1, 'r--', marker='o', markersize=4, label='PINN Solution')
    
    # 计算并显示误差作为标题
    error = np.max(np.abs(u_pinn_t1 - u_mat_t1))
    plt.title(f'Comparison at t ≈ {target_t:.2f} (Max Abs Error: {error:.4f})')
    
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True, linestyle=':')
    plt.legend()

    fig3 = plt.figure(figsize=(5, 5))
    dis=np.abs(u_mat_t1-u_pinn_t1)
    plt.plot(x_mat_vec, dis, 'k-', linewidth=2, label='error')
    plt.title(f'Comparison at t ≈ {target_t:.2f} (Max Abs Error: {error:.4f})')

    plt.xlabel('x')
    plt.ylabel('diff')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    nu = 0.0
    x = 2.0
    t = 3.0
    grid = 100
    training_times=3500
    #thread_matlab = threading.Thread(target=matlab_accsol, args=(nu, x, t, grid))
    #thread_matlab.start()

    thread_PINN = threading.Thread(target=PINN, args=(nu, x, t, grid,training_times))
    thread_PINN.start()

    #thread_matlab.join()
    thread_PINN.join()
    print("Max error: ")
    compare()
