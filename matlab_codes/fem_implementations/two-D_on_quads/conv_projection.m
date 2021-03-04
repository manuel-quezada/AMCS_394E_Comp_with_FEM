% In this code we project a function f(x) to the 1D pw linear FE space
clc; clear all
close all; hold on

% ***** DEFINE GENERAL PARAMETERS ***** %
%u = @(x,y) x;
u = @(x,y) sin(2*pi*x).^4 .* sin(2*pi*y).^4;
%u = @(x,y) 1.0*(x>=0.35).*(x<=0.65).*(y>=0.35).*(y<=0.65);
Nq = 2;

Nrows=4; % number of rows in the conv table
Nx=10; % number of cells for the first experiment
Ny=10;

conv = zeros(Nrows,5); % table for convergence study
for c=1:Nrows
    % ***** CREATE THE MESH ***** %
    C = zeros(Nx*Ny,4);
    coord_nodes = zeros((Nx+1)*(Ny+1),2);
    x = linspace(0,1,Nx+1);
    y = linspace(0,1,Ny+1);
    [yy,xx] = meshgrid(y,x);
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
    
    % ***** DO THE PROJECTION ***** %
    [U,M,ML] = projection_2D(u,coord_nodes,C,Nq,false,false);

    % ***** PLOT ***** %
    UU=xx*0;
    for j=1:Ny+1
        UU(1:(Nx+1),j) = U((j-1)*(Nx+1)+1:(j-1)*(Nx+1)+(Nx+1));
    end
    clf
    surf(xx,yy,UU)
    pause(1)
    
    % ***** GET ERRORS ***** %
    Nel = Nx*Ny;
    [E1,E2] = get_errors(u,U,coord_nodes,C,Nq);
    conv(c,1) = Nel;
    conv(c,2) = E1;
    conv(c,4) = E2;
    if c>1
        conv(c,3) = log(E1/conv(c-1,2))/log(0.5);
        conv(c,5) = log(E2/conv(c-1,4))/log(0.5);
    end
    
    % Refine the mesh
    Nx = Nx*2;
    Ny = Ny*2;
end
format short g
conv

