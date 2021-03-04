function [nodes,weights] = quadrature(Nq)
% This function computes the quad points and weights
% INPUT PARAMETERS:
%    Nq: number of quad points in 1D

a=0; b=1;
m=0.5*(a+b);
d=b-a;

switch Nq
    case 1
        nodes_1D=[m];
        weights_1D=[d];            
    case 2
        aux=d/2*sqrt(3)/3;
        nodes_1D=[m-aux,m+aux];
        weights_1D=[d/2,d/2];        
    case 3
        aux=d/2*sqrt(3/5);
        nodes_1D=[m-aux,m,m+aux];
        weights_1D=[5/18,8/18,5/18]*d;
    case 4
        aux_pos=d/2*sqrt(1/35*(15+2*sqrt(30)));
        aux_neg=d/2*sqrt(1/35*(15-2*sqrt(30)));
        nodes_1D=[m-aux_pos,m-aux_neg,m+aux_neg,m+aux_pos];
        weights_1D=[1/4-1/12*sqrt(5/6),1/4+1/12*sqrt(5/6),1/4+1/12*sqrt(5/6),1/4-1/12*sqrt(5/6)]*d;        
end

x_nodes=zeros(Nq*Nq,1);
y_nodes=zeros(Nq*Nq,1);
weights=zeros(Nq*Nq,1);
for j=1:Nq
    for i=1:Nq
        ij = (j-1)*Nq + i;
        x_nodes(ij) = nodes_1D(i);
        y_nodes(ij) = nodes_1D(j);
        weights(ij) = weights_1D(i)*weights_1D(j);
    end
end
nodes = [x_nodes, y_nodes];   


