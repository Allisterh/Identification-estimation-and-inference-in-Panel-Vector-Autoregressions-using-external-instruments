function [G, Gcum] = g_matrices(AL, C, p, hori, n)
% Computes derivatives of vec(C) wrt vec(A) via Lütkepohl's formula.
% AL: (n x np), C: (n x n*hori), p: lag order, hori: horizon, n: vars.
% Returns G and Gcum, each (n^2 x n^2*p x hori+1).
J   = [eye(n), zeros(n,(p-1)*n)];
Alut = [AL; eye(n*(p-1)), zeros(n*(p-1),n)];
AJ  = zeros(n*p, n, hori);
for k = 1:hori
    AJ(:,:,k) = (Alut^(k-1)) * J';
end
JAp = reshape(AJ, [n*p, n*hori])';
AJaux = zeros(size(JAp,1)*n, size(JAp,2)*n, hori);
Caux  = reshape([eye(n), C(:,1:(hori-1)*n)], [n,n,hori]);
for i = 1:hori
    AJaux(((n^2)*(i-1))+1:end, :, i) = kron(JAp(1:n*(hori+1-i),:), Caux(:,:,i));
end
Gaux = permute(reshape(sum(AJaux,3)', [(n^2)*p, n^2, hori]), [2,1,3]);
G = zeros(size(Gaux,1), size(Gaux,2), size(Gaux,3)+1);
G(:,:,2:end) = Gaux;
Gcum = cumsum(G, 3);
end
