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
    eng.burgers(nu, x, t, grid, nargout=0)


def PINN(nu: float, x: float, t: float, grid: int,training_times:int) -> None:
    print("PINN")
    burgers_PINN.sol(nu, x, t, grid,training_times)


def compare():
    
    mat_path = os.path.join(current_dir, 'burgers_matlab.mat')
    mat = scipy.io.loadmat(mat_path)
    u_mat = mat['u'].T
    PINN_path = os.path.join(current_dir, 'burgers_PINN.npy')
    data = np.load(PINN_path, allow_pickle=True).item()
    u_PINN = data.get('u')
    x_PINN = data.get('x')
    t_PINN = data.get('t')

    print(np.max(np.abs(u_PINN-u_mat)))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    dis=np.abs(u_PINN-u_mat)
    surf = ax.plot_surface(x_PINN,t_PINN,dis, cmap=plt.cm.cividis)
    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('t', labelpad=20)
    ax.set_zlabel('u', labelpad=20)
    fig.suptitle(f'Max error:{np.max(np.abs(u_PINN-u_mat))}')
    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.show()


if __name__ == '__main__':
    nu = 0.01
    x = 2.0
    t = 3.0
    grid = 100
    # training_times=5000
    # thread_matlab = threading.Thread(target=matlab_accsol, args=(nu, x, t, grid))
    # thread_matlab.start()

    # thread_PINN = threading.Thread(target=PINN, args=(nu, x, t, grid,training_times))
    # thread_PINN.start()

    # thread_matlab.join()
    # thread_PINN.join()
    print("Max error: ")
    compare()
