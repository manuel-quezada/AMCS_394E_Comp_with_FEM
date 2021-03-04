% In this code we solve:
% -\Delta u = f(x), \forall x\in[0,1]^2 
% u(x)=0 on the boundary
% f(x)=...(see step-40 in deal.ii)
clc; clear all
clf; close all; hold on

% ************************************* %
% ***** DEFINE GENERAL PARAMETERS ***** %
% ************************************* %
f = @(x,y) 1.0*(y>0.5+0.25*sin(4*pi*x)) - 1.0*(y<=0.5+0.25*sin(4*pi*x));
Nq = 2;
dofs_per_cell = 4;

Nx=80; % number of cells in each dimension
Ny=80;

% *************************** %
% ***** CREATE THE MESH ***** %
% *************************** %
C = zeros(Nx*Ny,4);
Nel = size(C,1); % rows of connectivity matrix
coord_nodes = zeros((Nx+1)*(Ny+1),2);
x = linspace(0,1,Nx+1);
y = linspace(0,1,Ny+1);
[yy,xx] = meshgrid(y,x);

% boundary nodes
Nh=size(coord_nodes,1);
bottom_boundary = 1:Nx+1;
top_boundary = Nh-Nx:Nh;
left_boundary = 1:Nx+1:Nh-Nx;
right_boundary = Nx+1:Nx+1:Nh;

% nodes
for j=1:Ny+1
    coord_nodes((j-1)*(Nx+1)+1:(j-1)*(Nx+1)+Nx+1,1)=x;
    coord_nodes((j-1)*(Nx+1)+1:(j-1)*(Nx+1)+Nx+1,2)=y(j);        
end
% conectivity matrix
for Ky=1:Ny
    for Kx=1:Nx
        C((Ky-1)*Nx + Kx,1) = (Ky-1)*(Nx+1) + Kx;
        C((Ky-1)*Nx + Kx,2) = (Ky-1)*(Nx+1) + Kx+1;
        C((Ky-1)*Nx + Kx,3) = Ky*(Nx+1) + Kx;
        C((Ky-1)*Nx + Kx,4) = Ky*(Nx+1) + Kx+1;            
    end
end

% *************************** %
% ***** SHAPE FUNCTIONS ***** %
% *************************** %
[shape,dxshape,dyshape] = Q_shape_functions(2);

% **************************** %
% ***** GLOBAL OPERATORS ***** %
% **************************** %
Nh=size(coord_nodes,1); % number of nodes (remember that p=1)
S = zeros(Nh,Nh);
F = zeros(Nh,1);

% *************************** %
% ***** QUADRATURE RULE ***** %
% *************************** %
[quad_points,quad_weights]=quadrature(Nq);

% ******************************* %
% ***** FINITE ELEMENT LOOP ***** %
% ******************************* %
% Create element based data structures %
Se=zeros(dofs_per_cell,dofs_per_cell);
Fe=zeros(dofs_per_cell,1);
grad_shape=zeros(2,dofs_per_cell);
for K=1:Nel % loop on cells
    % init to zero element based operators
    Se=Se*0;
    Fe=Fe*0;
    
    % get local dof indices
    local_dof_indices = C(K,:);
    
    % get element transformation
    [xT,yT,J] = get_el_transformation(shape,dxshape,dyshape,coord_nodes,local_dof_indices);
    detJ = det(J);
    invJ = 1/det(J)*[[J(2,2), -J(1,2)]; [-J(2,1), J(1,1)]];
    
    % loop on quad points 
    for q=1:Nq*Nq
        wq = quad_weights(q);
        xq = quad_points(q,1);
        yq = quad_points(q,2);        
        detJ_x_dV = wq*detJ;
        xp = xT(xq,yq);
        yp = yT(xq,yq);
        
        % get grad of shape functions at quad points
        for i=1:dofs_per_cell
            grad_shape(:,i) = invJ * [dxshape{i}(xq,yq); dyshape{i}(xq,yq)];
        end

        % loop on i-DOFs to assemble Fe and MLe
        for i=1:dofs_per_cell       
            Fe(i) = Fe(i) + f(xp,yp)*shape{i}(xq,yq)*detJ_x_dV;
            
            % loop on j-DOFs to assemble Me
            for j=1:dofs_per_cell
                Se(i,j) = Se(i,j) + grad_shape(:,i)'*grad_shape(:,j)*detJ_x_dV;
            end
        end
    end   
    
    % assemble from local to global
    for i=1:dofs_per_cell
        ig = local_dof_indices(i);
        F(ig) = F(ig) + Fe(i);
        for j=1:dofs_per_cell
            jg = local_dof_indices(j);
            S(ig,jg) = S(ig,jg) + Se(i,j);
        end
    end
end

% *********************************** %
% ***** SET BOUNDARY CONDITIONS ***** %
% *********************************** %
boundary = [left_boundary, bottom_boundary, right_boundary, top_boundary];
S(boundary,:)=zeros(length(boundary),Nh);
for i=1:length(boundary)
    ig = boundary(i);
    S(ig,ig)=1;
end
F(boundary) = 0;

% **************************** %
% ***** SOLVE THE SYSTEM ***** %
% **************************** %
U=S\F;

% ***** PLOT ***** %
UU=xx*0;
for j=1:Ny+1
    UU(1:(Nx+1),j) = U((j-1)*(Nx+1)+1:(j-1)*(Nx+1)+(Nx+1));
end
clf
surf(xx,yy,UU)
xlabel('x')
ylabel('y')
view(30,20)

