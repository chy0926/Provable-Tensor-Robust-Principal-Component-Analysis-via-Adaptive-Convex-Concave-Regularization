function [X,E_hat,iter] = ATNN_RPCAsep_HOU(Ten, opts)
[m,n,p] = size(Ten);
if ~exist('opts','var')
    opts = [];
end
if isfield(opts,'r')
    r = opts.r;
else
    r = 10;
end
if isfield(opts,'maxIter')
    maxIter = opts.maxIter;
else
    maxIter = 500;
end
if isfield(opts,'rho')
    rho = opts.rho;
else
    rho = 1.05;%1.05
end
if isfield(opts,'tol')
    tol = opts.tol;
else
    tol = 1e-5;
end
if isfield(opts,'lambda')
    lambda = opts.lambda;
else
    lambda = 1/sqrt(max(m,n));
end

if isfield(opts,'prox_w'); w = opts.prox_w;
elseif isfield(opts,'w'); w = opts.w;
else
    error('opts must contain prox_tnn weight vector: prox_w or w');
end
if isfield(opts,'prox_P'); P = opts.prox_P;
elseif isfield(opts,'P'); P = opts.P;
else
    error('opts must contain prox_tnn parameter P');
end
if isfield(opts,'Slice_weights'); Slice_weights = opts.Slice_weights;
else
    error('opts must contain Slice_weights for prox_tnn');
end

% initialize
D = reshape(Ten,[m*n,p]);
Y = D;
[u_ini, s_ini,v_ini]=svd(D,'econ');
U_hat = u_ini(:,1:r)*s_ini(1:r,1:r);
V_hat = v_ini(:,1:r);
% compute spectral norm
norm_two = 0;
try
    norm_two = lansvd(Y, 1, 'L');
catch
    norm_two = svds(Y,1);
end
norm_inf = norm( Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;
A_hat = U_hat*V_hat';
E_hat = zeros( m*n, p);
mu = 1/norm_two;
mu_bar = mu * 1e7;

d_norm = norm(D, 'fro');

iter = 0;
total_svd = 0;
converged = false;
sv = 10;
while ~converged
    iter = iter + 1;

    % E update (unchanged)
    temp_T = D - A_hat + (1/mu)*Y;
    E_hat = max(temp_T - lambda/mu, 0);
    E_hat = E_hat + min(temp_T + lambda/mu, 0);

    % === B update: minimal replacement ===
    tmp = D - E_hat + (1/mu)*Y;  % M^(k)
    tmpU = tmp * V_hat;          % (mn) x r

    % Build N_tensor: m x n x r
    N_tensor = zeros(m,n,r);
    for i = 1:r
        N_tensor(:,:,i) = reshape(tmpU(:,i),[m,n]);
    end
        [B_tensor, tnn] = prox_tnnsep(N_tensor, 1/mu, w, P, Slice_weights);

    % Write back to U_hat (unchanged structure)
    for i = 1:r
        U_hat(:,i) = reshape(B_tensor(:,:,i), [m*n,1]);
    end
    obj = tnn + lambda*norm(E_hat(:),1);

    % V update (unchanged)
    [U,~,V] = svd(tmp'*U_hat, 'econ');
    V_hat = U*V';
    total_svd = total_svd + 1;
    A_hat = U_hat * V_hat';

    % Multiplier update (unchanged)
    Z = D - A_hat - E_hat;
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);

    % stop
    stopCriterion = norm(Z, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end

    if iter ==1 || mod(total_svd, 40) == 0
        disp([   '#svd ' num2str(total_svd) ' r(A) ' num2str(rank(A_hat))...
            ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
            ' 0bj ' num2str(obj)...
            ' stopCriterion ' num2str(stopCriterion)]);
    end

    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;
    end
end

X = reshape(A_hat,[m,n,p]);
E_hat = reshape(E_hat,[m,n,p]);
end
