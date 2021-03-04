function [U,M,ML] = projection_2D(u,coord_nodes,C,Nq,use_ML,use_dense_matrix)
% Project a function f(x) to the 2D pw linear FE space
% INPUT PARAMETERS:
%    u: exact solution u=@(x)... 
%    coord_nodes
%    C: connectivity matrix
%    Nq: number of quad points
%    use_ML: use lumped mass matrix? 

Nel = size(C,1); % rows of connectivity matrix
dofs_per_cell = 4;

% *************************** %
% ***** SHAPE FUNCTIONS ***** %
% *************************** %
[shape,dxshape,dyshape] = Q_shape_functions(2);

% **************************** %
% ***** GLOBAL OPERATORS ***** %
% **************************** %
Nh=size(coord_nodes,1); % number of nodes (remember that p=1)
if use_dense_matrix == true
    M = zeros(Nh,Nh);
else
    ISp = zeros(Nh*9,1);
    JSp = zeros(Nh*9,1);
    Sp = zeros(Nh*9,1);
end
F = zeros(Nh,1);
ML = zeros(Nh,1);

% *************************** %
% ***** QUADRATURE RULE ***** %
% *************************** %
[quad_points,quad_weights]=quadrature(Nq);

% Create element based data structures %
Me=zeros(dofs_per_cell,dofs_per_cell);
Fe=zeros(dofs_per_cell,1);
MLe=zeros(dofs_per_cell,1);

% ******************************* %
% ***** FINITE ELEMENT LOOP ***** %
% ******************************* %
%I=0;
aux=1;
for K=1:Nel % loop on cells
    % init to zero element based operators
    Me=Me*0;
    Fe=Fe*0;
    MLe=MLe*0;
    
    local_dof_indices = C(K,:);
    [xT,yT,J] = get_el_transformation(shape,dxshape,dyshape,coord_nodes,local_dof_indices);
    detJ = det(J);
    
    %Ie=0;
    % loop on quad points 
    for q=1:Nq*Nq
        wq = quad_weights(q);
        xq = quad_points(q,1);
        yq = quad_points(q,2);
        detJ_x_dV = wq*detJ;
        xp = xT(xq,yq);
        yp = yT(xq,yq);
        
        %Ie = Ie + detJ_x_dV;
        % loop on i-DOFs to assemble Fe and MLe
        for i=1:dofs_per_cell       
            Fe(i) = Fe(i) + u(xp,yp)*shape{i}(xq,yq)*detJ_x_dV;
            MLe(i) = MLe(i) + shape{i}(xq,yq)*detJ_x_dV;
            
            % loop on j-DOFs to assemble Me
            for j=1:dofs_per_cell
                Me(i,j) = Me(i,j) + shape{i}(xq,yq)*shape{j}(xq,yq)*detJ_x_dV;
            end
        end
    end
    %Ie
    %I=I+Ie;   
    
    % assemble from local to global
    for i=1:dofs_per_cell
        ig = local_dof_indices(i);
        F(ig) = F(ig) + Fe(i);
        ML(ig) = ML(ig) + MLe(i);
        for j=1:dofs_per_cell
            jg = local_dof_indices(j);
            if use_dense_matrix == true
                M(ig,jg) = M(ig,jg) + Me(i,j);
            else
                ISp(aux) = ig;
                JSp(aux) = jg;
                Sp(aux) = Me(i,j);
                aux = aux + 1;
            end
        end
    end
end
if use_dense_matrix == false
    M = sparse(ISp,JSp,Sp);
    %size(M)
    %spy(M)
    %length(nonzeros(M))
    %pause
end
% **************************** %
% ***** SOLVE THE SYSTEM ***** %
% **************************** %
if use_ML
    U=1./ML.*F;
else
    U = M\F;
end

