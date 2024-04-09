import torch
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StreamfunctionNetwork(nn.Module):
    def __init__(self, L):
        super().__init__()

        self.layer1 = nn.Linear(  2,  L ) #sin and cos for 3 inputs gives 6 dimensions
        self.layer2 = nn.Linear(  L,  int(L/2) ) 
        self.layer3 = nn.Linear(  int(L/2),  1 ) #single output is streamfunction
        
        # self.conv1 = nn.Conv1d( in_channels=1, out_channels=L, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        #force network to learn periodic functions of the input

        #linear + gaussian activation
        y = self.layer1(x)
        # y = torch.exp( -torch.square(y) )
        y = torch.tanh(y)

        #linear + gaussian activation        
        y = self.layer2(y)
        # y = torch.exp( -torch.square(y) )
        y = torch.tanh(y)

        #linear to get streamfunction
        y = self.layer3(y)

        return y
    
class HydroNetwork(nn.Module):
    def __init__(self, streamnetwork):
        super().__init__()
        self.streamnetwork = streamnetwork        

    def forward(self, x):
        psi = self.streamnetwork(x) #evaluate the streamfunction
        
        #autodiff the streamfunction
        dpsi = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        # flow is u \hat{i} + v \hat{j}
        u =  dpsi[:,1]
        v = -dpsi[:,0]

        # Compute the Laplacian (divergence of the gradient)
        # w = -lap(psi)
        w =-torch.autograd.grad(dpsi[:, 0], x, grad_outputs=torch.ones_like(dpsi[:, 0]), create_graph=True, retain_graph=True)[0][:, 0] \
           -torch.autograd.grad(dpsi[:, 1], x, grad_outputs=torch.ones_like(dpsi[:, 1]), create_graph=True, retain_graph=True )[0][:, 1]
        
        f = torch.stack( (w,u,v), dim=1 )

        return f
    
class WeakPINN(nn.Module):
    def __init__( self, hydro_model, nu, p ):
        super().__init__()
        self.model = hydro_model
        self.nu = nu


    def forward(self, xs, xs_uniform, hydro_model, true_data, grids, particle_data, frame):

        x_grid, y_grid = grids[0].requires_grad_(True), grids[1].requires_grad_(True)
        # [x,y] = torch.meshgrid( (x_grid, y_grid) )
        # [X,Y] = torch.meshgrid( (x_grid, y_grid) )
        
        [y,x] = torch.meshgrid( (y_grid, x_grid) )
        [Y,X] = torch.meshgrid( (y_grid, x_grid) )

        x = torch.reshape( x, [-1,1] )
        y = torch.reshape( y, [-1,1] )
        
        xs      = torch.cat( (x,y), dim=1 )
        
        f = self.model(xs)

        f = torch.reshape(f, [79, 44, 3])
        # f = torch.transpose(f, dim0=1, dim1=0)

        
        # Get the PIV measurements of the real flow (u, v)
        u_real, v_real = torch.tensor(true_data[0]).requires_grad_(False), torch.tensor(true_data[1]).requires_grad_(False)
        
        # # Do a different grid
        # x_grid_new, y_grid_new = torch.linspace(0, 351, 352, requires_grad=True), torch.linspace(0, 631, 632, requires_grad=True)
        # Y_new, X_new = torch.meshgrid( (y_grid_new, x_grid_new) )
        # X_new = torch.reshape( X_new, [-1,1] )
        # Y_new = torch.reshape( Y_new, [-1,1] )
        
        # xs_new = torch.cat( (X_new, Y_new), dim=1 )
        # f_new = hydro_model.forward( xs_new )
        # u_new = f_new[...,1] #x-component of the flow
        # v_new = f_new[...,2] #y-component of flow

        COM, particle1, particle2, particle3, particle4 = particle_data[:]
        radius = np.linalg.norm( abs( COM[0][1:] - particle1[0][1:] ) )
        
        x_conv, y_conv = 44/705, 79/1265  # conversion from PIVLab coords to pixel coords
        # x_conv, y_conv = 1/2, 1/2
        
        def interpolate_points(P1, P2, num_points=100):
            """Interpolates points between two points."""
            x_values = np.linspace(int(P1[1]*x_conv), int(P2[1]*x_conv), num_points)
            y_values = np.linspace(int(P1[2]*y_conv), int(P2[2]*y_conv), num_points)
            return x_values, y_values

        # Number of points to interpolate
        num_points = 100

        # Interpolate points along the lines
        x1, y1 = interpolate_points(particle4[frame], particle1[frame], num_points)  # points along arm 1
        x2, y2 = interpolate_points(particle3[frame], particle2[frame], num_points)  # points along arm 2

        # ##### Visualize #####
        # plt.quiver(Y[:, :].detach().numpy(), X[:, :].detach().numpy(), v_real.detach().numpy(), u_real.detach().numpy())
        
        # # Plot lines of dots
        # plt.scatter(y1, x1, color='red')  # Line 1
        # plt.scatter(y2, x2, color='red')   # Line 2
        
        # # plt.figure( figsize=(10,6) )
        # # plt.quiver(Y_new[:, :].detach().numpy(), X_new[:, :].detach().numpy(), v_new.detach().numpy(), u_new.detach().numpy())
        
        # plt.scatter(particle1[frame][2]*y_conv, particle1[frame][1]*x_conv, marker='o', color='b')  # blue dot at particle 1
        # plt.scatter(particle2[frame][2]*y_conv, particle2[frame][1]*x_conv, marker='o', color='b')  # blue dot at particle 2
        # plt.scatter(particle3[frame][2]*y_conv, particle3[frame][1]*x_conv, marker='o', color='b')  # blue dot at particle 3
        # plt.scatter(particle4[frame][2]*y_conv, particle4[frame][1]*x_conv, marker='o', color='b')  # blue dot at particle 4
        
        # # Lines connectign particles 1 and 4, and 2 and 3
        # plt.plot( [particle1[frame][2]*y_conv, particle4[frame][2]*y_conv], [particle1[frame][1]*x_conv, particle4[frame][1]*x_conv], color='b', lw=4 )
        # plt.plot( [particle2[frame][2]*y_conv, particle3[frame][2]*y_conv], [particle2[frame][1]*x_conv, particle3[frame][1]*x_conv], color='b', lw=4 )

        # # plt.plot( [COM[frame][1]*x_conv, COM[frame][1]*x_conv], [(COM[frame][2]-100)*y_conv, (COM[frame][2]+100)*y_conv], color='orange' )
        # # plt.plot( [COM[frame][2]*y_conv, COM[frame][2]*y_conv], [(COM[frame][1]-100)*x_conv, (COM[frame][1]+100)*x_conv], color='orange' )
        
        # plt.savefig(f'/Users/darinmomayezi/Desktop/images/input{frame}.png', dpi=300)
        # # plt.show()
        # plt.close()
        
        # Get the coordinates of the two lines forming the jack
        # To do no slip : find normal and tangential components of velocity of each arm
        # tangential component comes from COM motion, which is negligeable in this data, so neglect it here.
        
        # omega = pd.read_csv('/Users/darinmomayezi/Desktop/omega_data.csv')  # get angular velocity
        # dxdt, dydt = omega['dxdt'].to_numpy(), omega['dydt'].to_numpy()        
        

        
        # get indices between particles 1 and 4
        # print( int( particle1[frame][1]*x_conv ), int( particle1[frame][2]*y_conv ) )

        w = f[...,0] #vorticity
        u = f[...,1] #x-component of the flow
        v = f[...,2] #y-component of flow
        u = u[:, :]
        v = v[:, :]
        
        # # Visualize
        # plt.quiver(Y[:, :].detach().numpy(), X[:, :].detach().numpy(), u.clone().detach().numpy(), v.clone().detach().numpy())
        # plt.show()
        
        
        # particles 1 and 4 make a side and particles 2 and 3 make the other side of the jack
        
        
        err = torch.abs( u.flatten() - u_real.flatten() ) + torch.abs( v.flatten() - v_real.flatten() )

        return err