function [shape,dxshape,dyshape] = Q_shape_functions(dim)
% compute the shape functions for p=1 and dim={1,2}

% Q1 shape functions in 1D 
phi0 = @(x) 1-x;
phi1 = @(x) x;
dphi0 = @(x) -1 + x*0;
dphi1 = @(x)  1 + x*0;

if dim==1
    shape = {phi0, phi1};
    dxshape = {dphi0, dphi1};
    dyshape = dxshape;
else
    % Q1 shape functions in 
    pphi0 = @(x,y) phi0(x).*phi0(y);
    pphi1 = @(x,y) phi1(x).*phi0(y);
    pphi2 = @(x,y) phi0(x).*phi1(y);
    pphi3 = @(x,y) phi1(x).*phi1(y);
        
    dxpphi0 = @(x,y) dphi0(x).*phi0(y);
    dxpphi1 = @(x,y) dphi1(x).*phi0(y);
    dxpphi2 = @(x,y) dphi0(x).*phi1(y);
    dxpphi3 = @(x,y) dphi1(x).*phi1(y);
        
    dypphi0 = @(x,y) phi0(x).*dphi0(y);
    dypphi1 = @(x,y) phi1(x).*dphi0(y);
    dypphi2 = @(x,y) phi0(x).*dphi1(y);
    dypphi3 = @(x,y) phi1(x).*dphi1(y);        
        
    shape = {pphi0,pphi1,pphi2,pphi3};
    dxshape = {dxpphi0,dxpphi1,dxpphi2,dxpphi3};        
    dyshape = {dypphi0,dypphi1,dypphi2,dypphi3};                
end
