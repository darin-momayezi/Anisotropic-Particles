import torch
from torch import nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StreamfunctionNetwork(nn.Module):
    def __init__(self, L):
        super().__init__()

        self.layer1 = nn.Linear(  3,  L ) #sin and cos for 3 inputs gives 6 dimensions
        self.batchnorm1 = nn.BatchNorm1d(L)
        self.layer2 = nn.Linear(  L,  int(L/2) ) 
        self.batchnorm2 = nn.BatchNorm1d(int(L/2))
        self.layer3 = nn.Linear(  int(L/2),  1 ) #single output is streamfunction
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
        
        psi = self.streamnetwork(x) #evaluate the streamfunction
        
        #autodiff the streamfunction
        dpsi = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        
        u =  dpsi[:,1]
        v = -dpsi[:,0]
        
        w = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(dpsi[:, 0]), create_graph=True, retain_graph=True)[0][:, 0] \
           -torch.autograd.grad(u, x, grad_outputs=torch.ones_like(dpsi[:, 1]), create_graph=True, retain_graph=True )[0][:, 1]
           
        dw = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)

        lapw = torch.autograd.grad(dw[0][:,0], x, grad_outputs=torch.ones_like(dw[0][:,0]), create_graph=True, retain_graph=True)[0][:, 0] +\
               torch.autograd.grad(dw[0][:,1], x, grad_outputs=torch.ones_like(dw[0][:,0]), create_graph=True, retain_graph=True)[0][:, 1]
        
        # Product rule : div dot (uw) = w_i * del_i * u_i + u_i * del_i * w_i
        # div_dot_uw = torch.mul(w, du[:,0]) + torch.mul(w, dv[:,1]) + torch.mul(u, dw[0][:,0]) + torch.mul(v, dw[0][:,1])
        
        divw_dot_u = torch.mul(dw[0][:,0], u) + torch.mul(dw[0][:,1], v)
        
        dwdx, dwdy, dwdt = dw[0][:, 0], dw[0][:, 1], dw[0][:, 2]
        NSE = dwdt + divw_dot_u - self.nu*lapw  # enforce NSE vorticity equation : dw/dt + del x (w x u) = nu*lap(w)
            
        f = torch.stack( (u,v), dim=1 )

        return f, NSE
    
class WeakPINN(nn.Module):
    def __init__( self, hydro_model, nu ):
        super().__init__()
        self.model = hydro_model
        self.nu = nu


    def forward(self, xs, true_data, grids, epoch, epochs):
        
        x_grid, y_grid, t_grid = grids[0].requires_grad_(False), grids[1].requires_grad_(False), grids[2].requires_grad_(False)
        [X,Y,T] = torch.meshgrid( (x_grid, y_grid, t_grid) )
        
        u_real, v_real = true_data[:]
        
        f, NSE = self.model(xs)
        ns=( len(x_grid), len(y_grid),len(t_grid) )
        f = torch.reshape(f, [ns[0], ns[1], ns[2], 2])
        
        u = f[...,0] #x-component of the flow
        v = f[...,1] #y-component of flow
        u = u[:, :]
        v = v[:, :]
        
        # err = torch.abs( u[:, :, :frames].flatten() - u_real[:, :].detach().numpy().flatten() ) + torch.abs( v[:, :, :frames].flatten() - v_real[:, :].detach().numpy().flatten() )
        
        # NSE = dwdt + div_dot_uw - self.nu*lapw  # enforce NSE vorticity equation : dw/dt + del x (w x u) = nu*lap(w)

        # note : del x (w x u) becomes del dot (uw) in 2D
 
        if epoch == epochs-1:
            for frame in range(int(v.shape[-1])):
                plt.quiver(X[...,0].detach().numpy(), Y[...,0].detach().numpy(), u[...,frame].detach().numpy(), v[...,frame].detach().numpy())
                plt.savefig(f'/Users/darinmomayezi/Documents/Research/SchatzLab/Codes/2DPINNs/New Data/2/images/reconstruction{frame}.png', dpi=300)
                plt.close('all')
            
        return NSE, u, v