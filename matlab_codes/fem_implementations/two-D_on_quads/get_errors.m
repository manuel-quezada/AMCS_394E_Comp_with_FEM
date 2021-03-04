function [E1,E2] = get_errors(u,U,coord_nodes,C,Nq)
% This function computes E1=int(|f(x)-fh(x)|) and E2=sqrt(int((f(x)-fh(x))^2))
% Input parameters:
%    coord_nodes
%    u: exact solution u=@(x)... 
%    U: DoFs
%    C: connectivity matrix
%    Nq: number of quad points

% *************************** %
% ***** QUADRATURE RULE ***** %
% *************************** %
[quad_points,quad_weights]=quadrature(Nq);

% *************************** %
% ***** SHAPE FUNCTIONS ***** %
% *************************** %
[shape,dxshape,dyshape] = Q_shape_functions(2);

Nel=size(C,1);
E1=0;
E2=0;
for K=1:Nel % loop on cells
    % get local dofs
    local_dof_indices = C(K,:);
    
    % get transformation
    [xT,yT,J] = get_el_transformation(shape,dxshape,dyshape,coord_nodes,local_dof_indices);
    detJ = det(J);
    
    % get local dofs
    U_local_dofs = U(local_dof_indices);

    % loop on quad points
    E1e=0;
    E2e=0;
    for q=1:Nq*Nq
        % get quad points
        wq = quad_weights(q);
        xq = quad_points(q,1);
        yq = quad_points(q,2);
        
        xp = xT(xq,yq);
        yp = yT(xq,yq);
        detJ_x_dV = wq*detJ;
        
        % get shape functions at quad points
        shape_functions_at_xq = [shape{1}(xq,yq), shape{2}(xq,yq), shape{3}(xq,yq), shape{4}(xq,yq)]';
        % get solution uh at quad point
        uh_at_xq = U_local_dofs' * shape_functions_at_xq;
        
        % compute the errors
        E1e = E1e + abs(u(xp,yp)-uh_at_xq) * detJ_x_dV;
        E2e = E2e + (u(xp,yp) - uh_at_xq)^2 * detJ_x_dV;
        
    end
    E1 = E1 + E1e;
    E2 = E2 + E2e;
    %pause
end
E2 = sqrt(E2);
