function C = ma_rep(AL, p, hori)
% Transforms VAR coefficients AL (n x np) into MA coefficients C (n x n*hori).
n         = size(AL,1);
vecAL     = reshape(AL,[n,n,p]);
vecALrevT = zeros(n,n,hori);
for ihori = 1:hori
    if ihori < (hori-p)+1
        vecALrevT(:,:,ihori) = zeros(n,n);
    else
        vecALrevT(:,:,ihori) = vecAL(:,:,(hori-ihori)+1)';
    end
end
vecALrevT = reshape(vecALrevT,[n,n*hori]);
C = repmat(vecAL(:,:,1),[1,hori]);
for ihori = 1:hori-1
    C(:,(n*ihori)+1:(n*(ihori+1))) = [eye(n),C(:,1:n*ihori)] ...
        * vecALrevT(:,(hori*n-(n*(ihori+1)))+1:end)';
end
end
