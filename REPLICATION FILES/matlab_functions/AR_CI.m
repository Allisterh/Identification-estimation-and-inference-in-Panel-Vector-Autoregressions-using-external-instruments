function[MSWlbound,MSWubound,irf,WHat]=AR_CI(B,T,n,horizons,e,AR,p,Xtot,Z,eta,newey_lags,nvar,critval,scale)

% map AR into MA
Caux        = [eye(n),MARep(AR,p,horizons)]; 
C           = reshape(Caux,[n,n,horizons+1]);
[WHataux,WHat,V] = CovAhat_Sigmahat_Gamma(p,Xtot,Z,eta,newey_lags);
W1          = WHat(1:(n^2)*p,1:(n^2)*p);

W12         = WHat(1:(n^2)*p,1+(n^2)*p:end);

W2          = WHat(1+(n^2)*p:end,1+(n^2)*p:end);
[G,Gcum]    = Gmatrices(AR,MARep(AR,p,horizons),p,horizons,n); 
ahat      = zeros(n,horizons+1); 

bhat      = zeros(n,horizons+1);

chat      = zeros(n,horizons+1);

Deltahat  = zeros(n,horizons+1);
    
MSWlbound = zeros(n,horizons+1);

MSWubound = zeros(n,horizons+1);

casedummy = zeros(n,horizons+1);

for j =1:n
    
    for ih=1:horizons+1
        
    ahat(j,ih)     = (T*(B(nvar,1)^2))-(critval*W2(nvar,nvar));
    
    bhat(j,ih)     = -2*T*scale*(e(:,j)'*C(:,:,ih)*B)*B(nvar,1)...
        + 2*critval*scale*(kron(B',e(:,j)'))*G(:,:,ih)*W12(:,nvar)...
        + 2*critval*scale*e(:,j)'*C(:,:,ih)*W2(:,nvar);
    
    chat(j,ih)     = ((T^.5)*scale*e(:,j)'*C(:,:,ih)*B).^2 ...
        -critval*(scale^2)*(kron(B',e(:,j)'))*G(:,:,ih)*W1*...
        ((kron(B',e(:,j)'))*G(:,:,ih))' ...
        -2*critval*(scale^2)*(kron(B',e(:,j)'))*G(:,:,ih)*W12*C(:,:,ih)'*e(:,j)...
        -critval*(scale^2)*e(:,j)'*C(:,:,ih)*W2*C(:,:,ih)'*e(:,j); 
    
    Deltahat(j,ih) = bhat(j,ih).^2-(4*ahat(j,ih)*chat(j,ih));
    
    if ahat(j,ih)>0 && Deltahat(j,ih)>0
        
   casedummy(j,ih) = 1;
   
   MSWlbound(j,ih) = (-bhat(j,ih) - (Deltahat(j,ih)^.5))/(2*ahat(j,ih));
        
   MSWubound(j,ih) = (-bhat(j,ih) + (Deltahat(j,ih)^.5))/(2*ahat(j,ih));
        
    elseif ahat(j,ih)<0 && Deltahat(j,ih)>0
        
   casedummy(j,ih) = 2;
   
   MSWlbound(j,ih) = (-bhat(j,ih) + (Deltahat(j,ih)^.5))/(2*ahat(j,ih));
   
   MSWubound(j,ih) = (-bhat(j,ih) - (Deltahat(j,ih)^.5))/(2*ahat(j,ih));
   
    elseif ahat(j,ih)>0 && Deltahat(j,ih)<0
        
   casedummy(j,ih) = 3;
   
   MSWlbound(j,ih) = NaN;
   
   MSWubound(j,ih) = NaN;
   
    else 
        
   casedummy(j,ih) = 4;
   
   MSWlbound(j,ih) = -inf;
   
   MSWubound(j,ih) = inf;
   
    end
    
    end
end

MSWlbound(nvar,1)=scale;
    
MSWubound(nvar,1)=scale;
for ih = 1:horizons + 1
    
    for ivar = 1:n
        
        irf(ivar,ih)     = scale*e(:,ivar)'*C(:,:,ih)*B./B(nvar,1);
    end
end