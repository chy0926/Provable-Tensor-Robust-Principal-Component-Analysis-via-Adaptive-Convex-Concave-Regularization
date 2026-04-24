function [X,tnn,trank] = prox_tnnsep(Y,rho,w,P,Slice_weights)

[n1,n2,n3] = size(Y);

X = zeros(n1,n2,n3);
Y = fft(Y,[],3);
tnn = 0;
trank = 0;
  
rho = rho*w;

[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S);
r = P+length(find(S(P+1:end)-Slice_weights(1)*rho(P+1:end)>0));

if r>=1
    S = S(1:r)-Slice_weights(1)*rho(1:r);
    X(:,:,1) = U(:,1:r)*diag(S)*V(:,1:r)';
    tnn = tnn+sum(S);
    trank = max(trank,r);
end

halfn3 = round(n3/2);

for i = 2 : halfn3
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    r = P+length(find(S(P+1:end)-Slice_weights(i)*rho(P+1:end)>0));
    if r>=1
        S = S(1:r)-Slice_weights(i)*rho(1:r);
        X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
        tnn = tnn+sum(S)*2;
        trank = max(trank,r);
    end
    X(:,:,n3+2-i) = conj(X(:,:,i));
end

% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S);
    % r = length(find(S-rho>0));
    r = P+length(find(S(P+1:end)-Slice_weights(i)*rho(P+1:end)>0));
    if r>=1
        S = S(1:r)-Slice_weights(i)*rho(1:r);
        X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
        tnn = tnn+sum(S);
        trank = max(trank,r);
    end
end
tnn = tnn/n3;
X = ifft(X,[],3);
