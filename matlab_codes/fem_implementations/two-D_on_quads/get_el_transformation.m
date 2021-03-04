function [xT,yT,J] = get_el_transformation(shape,dxshape,dyshape,coord_nodes,local_dofs_indices)
% This function gets the element trans from ref to physical and the Jacobian
% INPUT PARAMETERS:
%    shape, dxshape, dyshape: shape functions in KHat
%    coord_nodes
%    local_dofs_indices: local to global index map for a given element

% get transformation
xT = @(xHat,yHat) 0*xHat*yHat;
yT = @(xHat,yHat) 0*xHat*yHat;

% get derivatives wrt xHat and yHat
dxdxHat = @(xHat,yHat) 0*xHat*yHat;
dydxHat = @(xHat,yHat) 0*xHat*yHat;
dxdyHat = @(xHat,yHat) 0*xHat*yHat;
dydyHat = @(xHat,yHat) 0*xHat*yHat;

dofs_per_cell = 4;
for i=1:dofs_per_cell
    ig = local_dofs_indices(i);
    xT = @(xHat,yHat) xT(xHat,yHat) + shape{i}(xHat,yHat)*coord_nodes(ig,1);
    yT = @(xHat,yHat) yT(xHat,yHat) + shape{i}(xHat,yHat)*coord_nodes(ig,2);

    dxdxHat = @(xHat,yHat) dxdxHat(xHat,yHat) + dxshape{i}(xHat,yHat)*coord_nodes(ig,1);
    dydxHat = @(xHat,yHat) dydxHat(xHat,yHat) + dxshape{i}(xHat,yHat)*coord_nodes(ig,2);
    dxdyHat = @(xHat,yHat) dxdyHat(xHat,yHat) + dyshape{i}(xHat,yHat)*coord_nodes(ig,1);
    dydyHat = @(xHat,yHat) dydyHat(xHat,yHat) + dyshape{i}(xHat,yHat)*coord_nodes(ig,2);
end

% get Jacobian
J = [[dxdxHat(0.5,0.5),dydxHat(0.5,0.5)]; [dxdyHat(0.5,0.5),dydyHat(0.5,0.5)]];  
