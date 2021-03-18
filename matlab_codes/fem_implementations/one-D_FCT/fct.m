clc; clf
% physical parameters 
T=1.0;
t=0.0;

% numerical parameters
dx=1E-2/1.0;
Nh=1/dx+1;
x=linspace(0,1,Nh)';
tol=1E-15;
dt=0.01*dx;

% initial condition 
%un=sin(2*pi*x).^12;
un=(x>=0.35).*(x<=0.65);
plot(x,un,'--r','linewidth',4)

% ***** Consistent Mass matrix ***** %
M=1/6*diag(ones(Nh-1,1),1) + 4/6*diag(ones(Nh,1),0) + 1/6*diag(ones(Nh-1,1),-1);
M(1,end-1)=1/6;
M(end,2)=1/6;
M=dx*M;

% ***** Lumped Mass matrix ***** %
ML=dx*diag(ones(Nh,1),0);
invML=(1/dx)*ones(Nh,1);

% ***** Transport matrix ***** %
c=diag(ones(Nh-1,1),1) - diag(ones(Nh-1,1),-1);
c(1,end-1)=-1;
c(end,2)=1;
c=1/2*c;

% ***** get dij matrix ***** %
dij = c*0;
for i=1:Nh
   for j=max(i-1,1):min(i+1,Nh)
       if (i~=j)
          dij(i,j) = max(0,max(c(i,j),c(j,i)));
       end
   end
   dij(i,i) = -sum(dij(i,:));
end

% create some data structures
MIN=min(un);
MAX=max(un);
unp1=un*0;
Rpos=un*0;
Rneg=un*0;
correction = un*0;
flux = dij*0;

% ***** loop in time ***** %
while (t<T)
    t
    % low- and high-order solutions
    uL = un - dt*invML.*(c-dij)*un;    
    uH = un - dt*invML.*c*un;    
    
    % aux quantities for FCT
    deltaU  = uL-un;

    % get flux_ij and other aux quantities
    for i=1:Nh
        mi = ML(i,i);
        Ui=un(i);        
        Ppos=0;
        Pneg=0;
        deltaUi = deltaU(i);
        umax = Ui;
        umin = Ui;
        for j=max(i-1,1):min(i+1,Nh)
            if (i~=j)
                Uj=un(j);
                umax = max(umax,Uj);
                umin = min(umin,Uj);
                deltaUj = deltaU(j);
                flux(i,j) = (ML(i,j)-M(i,j))*(deltaUj-deltaUi) - dt*dij(i,j)*(Uj-Ui);
                Ppos = Ppos + max(flux(i,j),0);
                Pneg = Pneg + min(flux(i,j),0);
            end
        end
        % get Q vectors
        uLi = max(umin,uL(i));
        uLi = min(umax,uLi);        
        Qpos = mi*(umax-uLi);
        Qneg = mi*(umin-uLi);

        % get R vectors
        if Ppos==0
            Rpos(i) = 1.0;
        else
            Rpos(i) = min(1,Qpos/Ppos);
        end
        if Pneg==0
            Rneg(i) = 1.0;
        else
            Rneg(i) = min(1,Qneg/Pneg);
        end
    end

    % verify that R vectors are between 0 and 1
    if (min(Rneg)<-tol || max(Rneg)>1.0+tol || min(Rpos)<-tol || max(Rpos)>1.0+tol)
        'error in computation of R vectors'
        [min(Rneg), max(Rneg)]        
        [min(Rpos), max(Rpos)]
        pause
    end
    
    % get limiters and flux correction
    for i=1:Nh
        correction(i)=0;
        for j=max(i-1,1):min(i+1,Nh)
            if (i~=j)                
                fij = flux(i,j);
                aij = min(Rpos(i),Rneg(j))*(fij>0) + min(Rneg(i),Rpos(j))*(fij<0);
                correction(i) = correction(i) + aij*fij;
                Ppos = Ppos + max(fij,0);
                Pneg = Pneg + min(fij,0);
            end
        end
        unp1(i) = uL(i) + 1/mi * correction(i);
    end
    
    % update time
    t=t+dt;
    un=unp1;  
end

% plot solution
hold on
plot(x,unp1,'-k','linewidth',2)
ylim([-0.5,1.5])
set(gca,'FontSize',40);


