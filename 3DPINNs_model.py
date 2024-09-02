import torch
from torch import nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StreamfunctionNetwork(nn.Module):
    def __init__(self, L):
        super().__init__()

        self.layer1 = nn.Linear(  4,  L ) #sin and cos for 3 inputs gives 6 dimensions
        self.batchnorm1 = nn.BatchNorm1d(L)
        self.layer2 = nn.Linear(  L,  int(L/2) ) 
        self.batchnorm2 = nn.BatchNorm1d(int(L/2))
        self.layer3 = nn.Linear(  int(L/2),  3 ) #single output is streamfunction
        # self.initialize_weights()
        
        
    def forward(self, x):
        
        y = self.layer1(x)
        y = self.batchnorm1(y)
        y = torch.sin(y)
        
        y = self.layer2(y)
        y = self.batchnorm2(y)
        y = torch.sin(y)
        
        y = self.layer3(y)

        return y
    
    def initialize_weights(self):
        nn.init.normal_(self.layer1.weight, mean=0.0, std=5.0)
        nn.init.normal_(self.layer2.weight, mean=0.0, std=5.0)
        nn.init.normal_(self.layer3.weight, mean=0.0, std=5.0)
    
class HydroNetwork(nn.Module):
    def __init__(self, streamnetwork):
        super().__init__()
        self.streamnetwork = streamnetwork
        self.nu = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        
        psi = self.streamnetwork(x) # evaluate the streamfunction
        
        dpsi = torch.zeros(psi.shape[0], psi.shape[1], x.shape[1])  # [29250, 3, 4]

        for i in range(psi.shape[1]):  # Loop over each component of psi
            dpsi[:, i, :] = torch.autograd.grad(
                psi[:, i], x, grad_outputs=torch.ones_like(psi[:, i]), create_graph=True
            )[0]
        
        u = dpsi[:,2,1] - dpsi[:,1,2]  # dpsi_z/dy - dpsi_y/dz
        v = dpsi[:,0,2] - dpsi[:,2,0]  # dpsi_x/dz - dpsi_z/dx
        w = dpsi[:,1,0] - dpsi[:,0,1]  # dpsi_y/dx - dpsi_x/dy
        
        du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        dv = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dw = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        
        omega_i = dw[:,2] - dv[:,1]  # dw/dy - dv/dz
        omega_j = du[:,2] - dw[:,0]  # du/dz - dw/dx
        omega_k = dv[:,0] - du[:,1]  # dv/dx - du/dy
        omega = torch.stack( (omega_i,omega_j,omega_k), dim=1 )
        
        domega = torch.zeros(omega.shape[0], omega.shape[1], x.shape[1])  # [29250, 3, 4]

        for i in range(omega.shape[1]):  # Loop over each component of psi
            domega[:, i, :] = torch.autograd.grad(
                omega[:, i], x, grad_outputs=torch.ones_like(omega[:, i]), create_graph=True
            )[0]
        
        d2omega_dx2 = torch.autograd.grad(domega[:,0,0], x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0]
        d2omega_dy2 = torch.autograd.grad(domega[:,1,1], x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,1]
        d2omega_dz2 = torch.autograd.grad(domega[:,2,2], x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,2]
        
        lapw = d2omega_dx2 + d2omega_dy2 + d2omega_dz2
        
        adv_i, adv_j, adv_k = torch.mul(u, du[:,0]), torch.mul(v, dv[:,1]), torch.mul(w, dw[:,2])
        
        NSE_i = domega[:,0,3] + adv_i - self.nu*lapw
        NSE_j = domega[:,1,3] + adv_j - self.nu*lapw
        NSE_k = domega[:,2,3] + adv_k - self.nu*lapw
        
        NSE = torch.stack( (NSE_i,NSE_j,NSE_k), dim=1 )  
        f = torch.stack( (u,v,w), dim=1 )

        return f, NSE
    
class WeakPINN(nn.Module):
    def __init__( self, hydro_model, nu ):
        super().__init__()
        self.model = hydro_model
        self.nu = nu


    def forward(self, xs, true_data, grids, epoch, epochs):
        
        x_grid, y_grid, z_grid, t_grid = grids[0].requires_grad_(False), grids[1].requires_grad_(False), grids[2].requires_grad_(False), grids[3].requires_grad_(False)
        [X,Y,Z,T] = torch.meshgrid( (x_grid, y_grid, z_grid, t_grid) )
        
        u1_real, v1_real, u2_real, v2_real = true_data[:]
        
        f, NSE = self.model(xs)
        ns=( len(x_grid),len(y_grid),len(z_grid),len(t_grid) )
        f = torch.reshape(f, [ns[0], ns[1], ns[2], ns[3], 3])
        
        u = f[...,0] #x-component of the flow
        v = f[...,1] #y-component of flow
        w = f[...,2]
        u = u[:, :]
        v = v[:, :]
        w = w[:, :]
        
        if epoch == epochs-1:
            for frame in range(int(v.shape[-1])):
                plt.quiver(X[:,6,:,frame].detach().numpy(), Z[:,6,:,frame].detach().numpy(), u[:,6,:,frame].detach().numpy(), w[:,6,:,frame].detach().numpy())
                plt.savefig(f'/Users/darinmomayezi/Documents/Research/SchatzLab/Codes/3DPINNs/Stokes Flow/restart/output_images/reconstruction_revised{frame}.png', dpi=300)
                plt.close('all')
            
        return NSE, u, v, w
