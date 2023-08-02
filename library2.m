%{
Generate a feature matrix G for vectors using simulated QED data
%}


%% Load data
clear;


% Calculate velocity field gradients

v = load('/Users/darinmomayezi/Documents/School/Spring 2023/SchatzLab/Data/Tracking/SymmetricJack1/velocity.mat');
z = load('/Users/darinmomayezi/Documents/School/Spring 2023/SchatzLab/Data/Tracking/SymmetricJack1/vorticity.mat');
v = load('/Users/darinmomayezi/Documents/School/Spring 2023/SchatzLab/Data/Tracking/SymmetricJack1/pivmeasurement.mat');
COM = load('/Users/darinmomayezi/Documents/School/Spring 2023/SchatzLab/Data/Tracking/COM_data.csv');
COM = COM(:, 2:3);
velocity = v.v_smoothed;  % Smoothed velocity field 
U = v.velocity_magnitude;
r = v.u_component;  % u-component of velocity
t = v.v_component;  % v-component of velocity

% Use smoothed velocities
% r = v.u_smoothed;
% t = v.v_smoothed;

new_r = [];  % velocity gradient
for index = 1:length(r)
    temp_r = r{index, 1};
    new_r(index, :, :) = temp_r;
end


new_t = [];  % velocity gradient
for index = 1:length(t)
    temp_t = t{index, 1};
    new_t(index, :, :) = temp_t;
end


velocity_gradient = cell(315, 1);  % Array for velocity gradient
% Calculate velocity gradient
for index = 1:315
    velocity_gradient_temp = gradient(velocity{index, 1}); 
    velocity_gradient{index, 1} = velocity_gradient_temp;
end

% convert cell/struct type to arrays
new_velocity = [];  % velocity field
for index = 1:length(velocity)
    temp_velocity = velocity{index, 1};
    new_velocity(index, :, :) = temp_velocity;
end

new_velocity_gradient = [];  % velocity gradient
for index = 1:length(velocity_gradient)
    temp_gradient = velocity_gradient{index, 1};
    new_velocity_gradient(index, :, :) = temp_gradient;
end

vorticity = v.vorticity;
new_vorticity = [];
for index = 1:length(vorticity)
    temp_vorticity = vorticity{index, 1};
    new_vorticity(index, :, :) = temp_vorticity;
end

divergence = v.divergence;
new_divergence = [];
for index = 1:length(divergence)
    temp_divergence = divergence{index, 1};
    new_divergence(index, :, :) = temp_divergence;
end

new_positions = [];
for index = 1:4
    positions = load(sprintf('/Users/darinmomayezi/Documents/School/Spring 2023/SchatzLab/Data/Tracking/SymmetricJack1/particle%d_data.csv', index));
    for index1 = 1:315
        new_positions(index1, :, index) = positions(index1, 2:3);
    end
end
Nt = 315;

% u = rand(Nt,2,4); % matrix of size [Nt, 2, 4] flow velocity
% w = rand(Nt,1,4); % matrix of size [Nt, 1, 4] vorticity
% A = rand(Nt,2,4); % matrix of size [Nt, 2, 4] symmetric trace-free velocity gradient
% D = rand(Nt,1,4); % matrix of size [Nt, 1, 4] symmetric velocity divergence
% x = rand(Nt,2,4); % matrix of size [Nt, 2, 4] positions
% dt= 1;  % timestep

u = new_velocity;
w = new_vorticity;
A = new_velocity_gradient;
D = new_divergence;
x = new_positions;
x = x-mean(x, 3);  % subtract the mean
dt = 1;

% Interpolation onto jack points
u_interp = [];  % shape = [time, flow components, jack points]
[X, Y] = meshgrid(1:44, 1:79);
for index = 1:315  % Iterating in time
    for particle = 1:4  % Iterating in jack arm
        x_pos = x(index, 1, particle);
        x_pos = reshape(x_pos, 1, 1);
        y_pos = x(index, 2, particle);
        y_pos = reshape(y_pos, 1, 1);
        pos = x(index, :, :);
        pos = reshape(pos, 2, 4);
        flow_t = u(index, :, :);  % Flow at time t
        flow_t = reshape(flow_t, 79, 44);
        interpolation1 = interp2(X, Y, flow_t, x_pos, y_pos);
        u_interp(index, :, particle) = interpolation1;
    end
end


r_interp = [];
[X, Y] = meshgrid(1:44, 1:79);
for index = 1:315
    for particle = 1:4
        x_pos = x(index, 1, particle);
        x_pos = reshape(x_pos, 1, 1);
        y_pos = x(index, 2, particle);
        y_pos = reshape(y_pos, 1, 1);
        pos = x(index, :, :);
        pos = reshape(pos, 2, 4);
        flow_t = new_r(index, :, :);  % u-component of flow at time t
        flow_t = reshape(flow_t, 79, 44);
        interpolation = interp2(X, Y, flow_t, x_pos, y_pos);
        if isnan(interpolation)
            interpolation = 0;
        end
        u_interp(index, 1, particle) = interpolation;
    end
end


t_interp = [];
[X, Y] = meshgrid(1:44, 1:79);
for index = 1:315
    for particle = 1:4
        x_pos = x(index, 1, particle);
        x_pos = reshape(x_pos, 1, 1);
        y_pos = x(index, 2, particle);
        y_pos = reshape(y_pos, 1, 1);
        pos = x(index, :, :);
        pos = reshape(pos, 2, 4);
        flow_t = new_t(index, :, :);  % t-component of flow at time t
        flow_t = reshape(flow_t, 79, 44);
        interpolation = interp2(X, Y, flow_t, x_pos, y_pos);
        if isnan(interpolation)
            interpolation = 0;
        end
        u_interp(index, 2, particle) = interpolation;
    end
end

u = u_interp;

% Line integral interpolation scheme
radii = [];
for time_step = 1:315
    rr = abs(new_positions(time_step, 1, 1) - new_positions(time_step, 1, 3));  % radius of jack
    radii(time_step) = rr;
    
end


% Interpolation using line integral along jack
radius = 70;
x_step = 720 / 44;
y_step = 890 / 79;
[X, Y] = meshgrid(1:x_step:720, 1:y_step:890);
M  = 64; %points on circle
theta = (0:(M-1))/M*2*pi; %theta
T = {};  % contains int dl*U
Q = {};  % contains contractions of int dl*grad*U
I = {};  % contains constractions of int dl*U*U
line_to_plot = [];
for time_step = 1:315
    x0 = COM(time_step, 1);  % COM x-coord
    y0 = COM(time_step, 2);  % COM y-coord
    x1 = new_positions(time_step, 1, 4);  % arm1 x-coord
    y1 = new_positions(time_step, 2, 4);  % arm1 y-coord
%     radius = sqrt( (x1 - x0)^2 + (y1 - y0)^2 ); 
    radius = 70;

    %x and y values of the circle
    x2 = x0 + radius * cos(theta);
    y2 = y0 + radius * sin(theta);

    line_to_plot(time_step, 1) = x1;
    line_to_plot(time_step, 2) = y1;
    xarm1 = line_to_plot(:,1);
    yarm1 = line_to_plot(:,2);

    U_sq = mean(r{time_step, 1}.^2 + t{time_step, 1}.^2, 'all');
    
%     hold on
%     quiver(X, Y, r{time_step, 1}/sqrt(U_sq), t{time_step, 1}/sqrt(U_sq), ...
%         'color', 'b')
%     quiver(x0,y0,x1-x0,y1-y0, 'color', 'g')
%     plot(x2, y2, 'color', 'r')
%     plot(xarm1, yarm1);
%     axis([0 719 0 889])
%     pbaspect([720/(720+890) 890/(720+890) 1])
% 
%     if time_step == 1
%         gif('/Users/darinmomayezi/Desktop/jack.gif')
%     elseif time_step > 1
%         gif  
%     end
% 
%     clf


    %interpolate flow (or its gradients) onto circle
    u2 = interp2(X,Y,normalize(r{time_step, 1}),x2,y2);  % x-component of U
    v2 = interp2(X,Y,normalize(t{time_step, 1}),x2,y2);  % y-component of U


    %compute dl = <dx,dy>
    dx = -y2./(x2.^2 + y2.^2);
    dy =  x2./(x2.^2 + y2.^2);

    integrate = @(field) [sum( field.*dx), sum(field.*dy) ]*2*pi/M;

    tt = zeros(2,2); %integral \int dl_i u_j tensor

    tt(:,1) = integrate(u2);
    tt(:,2) = integrate(v2);
    tt(isnan(tt)) = 0;

    T{time_step} = tt;

    % Tensor contractions for Q = int dl*grad*u
    q = zeros(3, 2);

    Ux = r{time_step, 1} / sqrt(U_sq);
    Uy = t{time_step, 1} / sqrt(U_sq);
    
    [gradx_Ux, grady_Ux] = gradient(Ux);
    [gradx_Uy, grady_Uy] = gradient(Uy);
    
    gradx_Ux = interp2(X,Y,gradx_Ux,x2,y2);
    gradx_Uy = interp2(X,Y,gradx_Uy,x2,y2);
    grady_Ux = interp2(X,Y,grady_Ux,x2,y2);
    grady_Uy = interp2(X,Y,grady_Uy,x2,y2);
    
    DgxUx = num2cell(integrate(gradx_Ux));
    DgxUy = num2cell(deal(integrate(gradx_Uy)));
    DgyUx = num2cell(integrate(grady_Ux));
    DgyUy = num2cell(integrate(grady_Uy));
    [dxgxUx, dygxUx] = DgxUx{:};
    [dxgxUy, dygxUy] = DgxUy{:};
    [dxgyUx, dygyUx] = DgyUx{:};
    [dxgyUy, dygyUy] = DgyUy{:};

    Q_k = [dxgxUx + dygyUx, dxgxUy + dygyUy];
    Q_j = [dxgxUx + dygxUy, dxgyUx + dygyUy];
    Q_i = [dxgxUx + dxgyUy, dygxUx + dygyUy];

    q(1, 1:2) = Q_i;
    q(2, 1:2) = Q_j;
    q(3, 1:2) = Q_k;

    Q{time_step} = q;

    % Tensor contractions for I = dl*U*U
    i = zeros(3,2);

    UU = {};  % Symmetric
    UxUx = interp2(X,Y,Ux.*Ux,x2,y2);  % element-wise multiplication ?
    UyUx = interp2(X,Y,Uy.*Ux,x2,y2);
    UxUy = interp2(X,Y,Ux.*Uy,x2,y2);
    UyUy = interp2(X,Y,Uy.*Uy,x2,y2);
    
    
    DUxUx = num2cell(integrate(UxUx));
    DUyUx = num2cell(integrate(UyUx));
    DUxUy = num2cell(integrate(UxUy));
    DUyUy = num2cell(integrate(UyUy));

    [dxUxUx, dyUxUx] = DUxUx{:};
    [dxUyUx, dyUyUx] = DUyUx{:};
    [dxUxUy, dyUxUy] = DUxUy{:};
    [dxUyUy, dyUyUy] = DUyUy{:};

    I_k = [dxUxUx + dyUyUx, dxUxUy + dyUyUy];
    I_j = [dxUxUx + dyUxUy, dxUyUx + dyUyUy];
    I_i = [dxUxUx + dxUyUy, dyUxUx + dyUyUy];

    i(1, 1:2) = I_i;
    i(2, 1:2) = I_j;
    i(3, 1:2) = I_k;

    I{time_step} = i;
end

new_T = [];
for time_step = 1:315
    component = T{time_step};
    new_T(time_step, 1:2, 1:2) = component;
end



new_Q = [];
for timestep = 1:315
    new_Q(timestep, 1:3, 1:2) = Q{timestep};
end

new_I = [];
for timestep = 1:315
    new_I(timestep, 1:3, 1:2) = I{timestep};
end

%Integrate library
addpath("../SPIDER_functions/");

number_of_library_terms = 3;   %under-estimate this
number_of_windows       = 128; %number of domains we integrate over 
degrees_of_freedom      = 2;   %vectors have one degree of freedom
dimension               = 1;   %how many dimensions does our data have?
envelope_power          = 4;   %weight is (1-x^2)^power
size_vec                = [32]; %how many gridpoints should we use per integration?


buffer                  = 0; %Don't use points this close to boundary

%define shorthand notation
nl = number_of_library_terms;
nw = number_of_windows;
dof= degrees_of_freedom;

%Make important objects for integration
pol      = envelope_pol( envelope_power, dimension );
G        = zeros( dof*nw, nl );
labels   = cell(nl, 1);
scales   = zeros(1,nl);

size_of_data = size(x, 1);
corners = pick_subdomains( size_of_data, size_vec, buffer, nw );


%Create grid variable
t = (1:size(x,1))*dt;
grid = {t};



%% actually integrating library
a = 1; %running index over library space

m = 1; %particle we investigate

% One term
labels{a} = "d/dt \vec{x}_1";
G(:,a)    = SPIDER_integrate_vector( COM(:,1,m), COM(:,2,m), [], grid, corners, size_vec, pol );
scales(a) = 1;
a         = a+1;

% COM velocity
labels{a} = "v";
G(:,a)    = SPIDER_integrate_vector( COM(:,1), COM(:,2), [1], grid, corners, size_vec, pol );
scales(a) = 1;
a         = a+1;

% Components of T
labels{a} = "T_1";
G(:,a)    = SPIDER_integrate_vector( new_T(:,1,1), new_T(:,1,2), [], grid, corners, size_vec, pol );
scales(a) = 1;
a         = a+1;

% Components of T
labels{a} = "T_2";
G(:,a)    = SPIDER_integrate_vector( new_T(:,2,1), new_T(:,2,2), [], grid, corners, size_vec, pol );
scales(a) = 1;
a         = a+1;

% Components of Q
labels{a} = "Q_1";
G(:,a)    = SPIDER_integrate_vector( new_Q(:,1,1), new_Q(:,1,2), [], grid, corners, size_vec, pol );
scales(a) = 1;
a         = a+1;

labels{a} = "Q_2";
G(:,a)    = SPIDER_integrate_vector( new_Q(:,2,1), new_Q(:,2,2), [], grid, corners, size_vec, pol );
scales(a) = 1;
a         = a+1;

labels{a} = "Q_3";
G(:,a)    = SPIDER_integrate_vector( new_Q(:,3,1), new_Q(:,3,2), [], grid, corners, size_vec, pol );
scales(a) = 1;
a         = a+1;

% Components of I

labels{a} = "I_1";
G(:,a)    = SPIDER_integrate_vector( new_I(:,1,1), new_I(:,1,2), [], grid, corners, size_vec, pol );
scales(a) = 1;
a         = a+1;

labels{a} = "I_2";
G(:,a)    = SPIDER_integrate_vector( new_I(:,2,1), new_I(:,2,2), [], grid, corners, size_vec, pol );
scales(a) = 1;
a         = a+1;

labels{a} = "I_3";
G(:,a)    = SPIDER_integrate_vector( new_I(:,3,1), new_I(:,3,2), [], grid, corners, size_vec, pol );
scales(a) = 1;
a         = a+1;

% for i = 1:79
%     for j = 1:4
%        labels{a} = "";
%         G(:,a)    = SPIDER_integrate_vector( u(:, 1, j), u(:, 2, j), [], grid, corners, size_vec, pol );
%         scales(a) = 1;
%         a         = a+1; 
% 
% 
%     end
% end


%% normalize with polynomial weight
norm_vec = SPIDER_integrate( 0*x(:,1,1) + 1, [], grid, corners, size_vec, pol );      
norm_vec = repmat( norm_vec, [dof,1] );

G = G./norm_vec;
G = G./scales;

G0 = G;

%% Split off time derivative

b = G(:,1);
G = G(:,2:end);

% alpha = 1;
% ATA = G.'*G;
% aDD = alpha*(del2(reshape(x, [], 2)));
% D2x = aDD(:, 1);
% D2y = aDD(:, 2);
% ATb = G.'*b;
% M = @(x) G.'*G*x - alpha*reshape(del2(reshape(x, [], 2)), 316, []);
% solution = gmres(M, G.'*b);

c = G\b;
residual = norm(G*c-b)/norm(b);
% c = reshape(c, [79, 44]);

% plot(asinh(c))
% norm(G*c-b) / norm(b);

function vals = SPIDER_integrate_vector( u, v, derivs, grid, corners, size_vec, pol )
  u = squeeze(u);
  v = squeeze(v);
  vals_1 = SPIDER_integrate( u, derivs, grid, corners, size_vec, pol );
  vals_2 = SPIDER_integrate( v, derivs, grid, corners, size_vec, pol );
  vals =[vals_1; vals_2];
end