function [L,S,obj,err,iter] = trpca_tnn_unifiedstop(X,lambda,opts)
% Original full-size CCSVS solver parameters are retained.
% Only the stopping quantity is replaced by the common relative primal
% residual: ||X-L-S||_F / max(||X||_F,eps).

if ~exist('opts','var'); opts = []; end

tol = 1e-8;
max_iter = 500;
rho = 1.1;
mu = 1e-4;
max_mu = 1e10;
DEBUG = 0;

if isfield(opts,'tol'); tol = opts.tol; end
if isfield(opts,'max_iter'); max_iter = opts.max_iter; end
if isfield(opts,'rho'); rho = opts.rho; end
if isfield(opts,'mu'); mu = opts.mu; end
if isfield(opts,'max_mu'); max_mu = opts.max_mu; end
if isfield(opts,'DEBUG'); DEBUG = opts.DEBUG; end
if isfield(opts,'w'); w = opts.w; else; error('opts.w is required'); end
if isfield(opts,'P'); P = opts.P; else; error('opts.P is required'); end
if isfield(opts,'Slice_weights'); Slice_weights = opts.Slice_weights;
else; error('opts.Slice_weights is required'); end

dim = size(X);
L = zeros(dim);
S = zeros(dim);
Y = zeros(dim);
x_norm = max(norm(X(:),'fro'),eps);

for iter = 1:max_iter
    [L,tnnL] = prox_tnn(-S+X-Y/mu,1/mu,w,P,Slice_weights);
    S = prox_l1(-L+X-Y/mu,lambda/mu);

    dY = L+S-X;
    stopCriterion = norm(dY(:),'fro')/x_norm;

    if DEBUG && (iter == 1 || mod(iter,10) == 0)
        obj = tnnL+lambda*norm(S(:),1);
        fprintf('iter %d, mu=%.6e, obj=%.6e, relRes=%.6e\n', ...
            iter,mu,obj,stopCriterion);
    end

    if stopCriterion < tol
        break;
    end

    Y = Y+mu*dY;
    mu = min(rho*mu,max_mu);
end

obj = tnnL+lambda*norm(S(:),1);
residualFinal = L+S-X;
err = norm(residualFinal(:));
end
