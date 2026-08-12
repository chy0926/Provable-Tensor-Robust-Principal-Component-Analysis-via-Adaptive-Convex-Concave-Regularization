function [X,E_hat,iter] = ATNN_RPCA_unifiedstop(Ten, opts)
% Original ATNN_RPCA with its intended stopping statement restored.
% Stopping rule: ||Y-L-E||_F / max(||Y||_F,eps) < opts.tol.

[m,n,p] = size(Ten);
if ~exist('opts','var'); opts = []; end
if isfield(opts,'r'); r = opts.r; else; r = 10; end
if isfield(opts,'maxIter'); maxIter = opts.maxIter; else; maxIter = 200; end
if isfield(opts,'rho'); rho = opts.rho; else; rho = 1.05; end
if isfield(opts,'tol'); tol = opts.tol; else; tol = 1e-5; end
if isfield(opts,'lambda'); lambda = opts.lambda; else; lambda = 1/sqrt(max(m,n)); end

Dobs = reshape(Ten,[m*n,p]);
Y = Dobs;
[u_ini,s_ini,v_ini] = svd(Dobs,'econ');
r = min([r,size(u_ini,2),size(v_ini,2)]);
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
    tnnL = 0;
    for i = 1:r
        [U,S,V] = svd(reshape(tmpU(:,i),[m,n]),'econ');
        diagS = diag(S);
        svp = nnz(diagS>1/mu);
        if svp > 0
            tnnL = tnnL+sum(diagS(1:svp));
            tmpU_hat = U(:,1:svp)*diag(diagS(1:svp)-1/mu)*V(:,1:svp)';
        else
            tmpU_hat = zeros(m,n);
        end
        U_hat(:,i) = tmpU_hat(:);
    end
    obj = tnnL+lambda*norm(E_hat(:),1);

    [Qu,~,Qv] = svd(tmp'*U_hat,'econ');
    V_hat = Qu*Qv';
    total_svd = total_svd+1;
    A_hat = U_hat*V_hat';

    Z = Dobs-A_hat-E_hat;
    Y = Y+mu*Z;
    mu = min(mu*rho,mu_bar);

    % Unified stopping rule; the original file had an empty if block here.
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
