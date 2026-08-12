function [X,E_hat,iter] = ATNN_RPCAsep_unifiedstop(Ten, opts)
% Original ATNN_RPCAsep_HOU structure with two controlled additions only:
%   (1) common relative primal-residual stopping rule;
%   (2) opts.learn_D switch for the fixed-transform ablation.

[m,n,p] = size(Ten);
if ~exist('opts','var'); opts = []; end

if isfield(opts,'r'); r = opts.r; else; r = 10; end
if isfield(opts,'maxIter'); maxIter = opts.maxIter; else; maxIter = 500; end
if isfield(opts,'rho'); rho = opts.rho; else; rho = 1.05; end
if isfield(opts,'tol'); tol = opts.tol; else; tol = 1e-5; end
if isfield(opts,'lambda'); lambda = opts.lambda; else; lambda = 1/sqrt(max(m,n)); end
if isfield(opts,'learn_D'); learn_D = logical(opts.learn_D); else; learn_D = true; end

if isfield(opts,'prox_w'); w = opts.prox_w;
elseif isfield(opts,'w'); w = opts.w;
else; error('opts must contain prox_w or w');
end
if isfield(opts,'prox_P'); P = opts.prox_P;
elseif isfield(opts,'P'); P = opts.P;
else; error('opts must contain prox_P or P');
end
if isfield(opts,'Slice_weights'); Slice_weights = opts.Slice_weights;
else; error('opts must contain Slice_weights');
end

Dobs = reshape(Ten,[m*n,p]);
Y = Dobs;
[u_ini,s_ini,v_ini] = svd(Dobs,'econ');
r = min([r, size(u_ini,2), size(v_ini,2)]);
U_hat = u_ini(:,1:r)*s_ini(1:r,1:r);
V_hat = v_ini(:,1:r);

try
    norm_two = lansvd(Y,1,'L');
catch
    norm_two = svds(Y,1);
end
norm_inf = norm(Y(:),inf)/lambda;
dual_norm = max(norm_two,norm_inf);
Y = Y/max(dual_norm,eps);

A_hat = U_hat*V_hat';
E_hat = zeros(m*n,p);
mu = 1/max(norm_two,eps);
mu_bar = mu*1e7;
d_norm = max(norm(Dobs,'fro'),eps);

iter = 0;
total_svd = 0;
converged = false;
while ~converged
    iter = iter+1;

    temp_T = Dobs-A_hat+(1/mu)*Y;
    E_hat = max(temp_T-lambda/mu,0)+min(temp_T+lambda/mu,0);

    tmp = Dobs-E_hat+(1/mu)*Y;
    tmpU = tmp*V_hat;
    N_tensor = zeros(m,n,r);
    for i = 1:r
        N_tensor(:,:,i) = reshape(tmpU(:,i),[m,n]);
    end
    [B_tensor,tnn] = prox_tnnsep(N_tensor,1/mu,w,P,Slice_weights);
    for i = 1:r
        U_hat(:,i) = reshape(B_tensor(:,:,i),[m*n,1]);
    end
    obj = tnn+lambda*norm(E_hat(:),1);

    if learn_D
        [Qu,~,Qv] = svd(tmp'*U_hat,'econ');
        V_hat = Qu*Qv';
    end
    total_svd = total_svd+1;
    A_hat = U_hat*V_hat';

    Z = Dobs-A_hat-E_hat;
    Y = Y+mu*Z;
    mu = min(mu*rho,mu_bar);

    % Unified stopping rule used by all five variants.
    stopCriterion = norm(Z,'fro')/d_norm;
    if stopCriterion < tol
        converged = true;
    end

    if iter == 1 || mod(total_svd,40) == 0
        fprintf('#iter %d r(A) %d |E|_0 %d obj %.6e relRes %.6e\n', ...
            iter,rank(A_hat),nnz(abs(E_hat)>0),obj,stopCriterion);
    end

    if ~converged && iter >= maxIter
        disp('Maximum iterations reached');
        converged = true;
    end
end

X = reshape(A_hat,[m,n,p]);
E_hat = reshape(E_hat,[m,n,p]);
end
