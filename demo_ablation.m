%% Demo: Controlled Ablation Study for AC^2-TRPCA
% Five variants required by the reviewer:
%   1) Full AC^2-TRPCA: learned D + reduced CCSVS
%   2) w/o concave compensation: learned D + reduced convex shrinkage (P = 0)
%   3) Fixed-D AC^2-TRPCA: fixed initial D + reduced CCSVS
%   4) ATNN: learned D + standard TNN
%   5) Full-size CCSVS: no dimension reduction
%
% This single script supports both hyperspectral images and videos.
% Required existing functions on the MATLAB path:
%   prox_tnnsep.m, prox_tnn.m, prox_l1.m, HSI_QA.m,
%   PseudoImage.m (optional), togetGif.m (optional), and lansvd or svds.
%
% MATLAB R2016b or later is required because local functions are included
% at the end of this script.

clc;
clear;
close all;

%% ========================= 1. Paths =========================
rootDir = fileparts(mfilename('fullpath'));
cd(rootDir);

addpath(genpath(fullfile(rootDir, 'lib')));
addpath(genpath(fullfile(rootDir, 'data')));
addpath(genpath(fullfile(rootDir, 'TC_Code')));
addpath(genpath(fullfile(rootDir, 'utils')));


%% ========================= 2. Data ==========================

dataName = 'pure_DCmall';
dataFile = fullfile(rootDir, 'data', dataName);  % .mat extension is optional

loadedData = load(dataFile);
Ohsi = extract_3d_tensor(loadedData);
Ohsi = double(Ohsi);

[height, width, band] = size(Ohsi);
ll = min(height, width);
fprintf('Data: %s, size = %d x %d x %d\n', dataName, height, width, band);

%% ====================== 3. Noise settings ===================
% noise_mode: 1 = sparse; 2 = Gaussian; 3 = Gaussian + sparse
noise_mode = 3;
val_sparse = 0.1;
val_gauss  = 0.1;
noise_seed = 42;

rng(noise_seed, 'twister');
Nhsi = add_test_noise(Ohsi, noise_mode, val_sparse, val_gauss);

%% ====================== 4. Common settings ==================

r3_common = 5;


P_common = 45;
if P_common > ll
    error('P_common=%d exceeds min(height,width)=%d.', P_common, ll);
end

% Common stopping criterion for all five variants:
% ||Y-L-S||_F / max(||Y||_F,eps) < tol_common
tol_common     = 1e-5;
maxIter_common = 500;
rho_common     = 1.05;
DEBUG           = 1;

% Regularization parameters.
lambda_reduced = 1 / sqrt(max(height, width));
lambda_full    = 1 / sqrt(max(height, width) * band);

% CCSVS weights. Keep the same P and the same spatial singular-value rule
% for the full proposed method, Fixed-D, and full-size CCSVS.
omega_top = linspace(1e-5, 1e-7, P_common)';
omega_tail = 0.7;
w_ccsvs = [omega_top; omega_tail * ones(ll - P_common, 1)];

% Removing the concave compensation term: P = 0 and ordinary shrinkage
% is applied to all singular values.
w_no_concave = omega_tail * ones(ll, 1);

% Slice weights. All reduced CCSVS variants must share the same vector.
slice_weights_reduced = [1, 1* ones(1, r3_common - 1)];
slice_weights_full    = [1, 1 * ones(1, band - 1)];

% Full-size CCSVS ADMM penalty initialization.
mu_full     = 1e-2;
max_mu_full = 1e10;

% Apply identical output clipping to all methods before evaluation.
clip_output = true;

%% ======================== 5. Run flags ======================
Run_Full_AC2       = true;
Run_No_Concave     = true;
Run_Fixed_D        = true;
Run_ATNN           = true;
Run_Full_CCSVS     = true;

methodName = { ...
    'Noisy', ...
    'Full AC2-TRPCA', ...
    'w/o Concave', ...
    'Fixed-D AC2', ...
    'ATNN', ...
    'Full-size CCSVS'};
Mnum = numel(methodName);

Results = cell(1, Mnum);
Time    = nan(1, Mnum);
Iter    = nan(1, Mnum);
MPSNR   = nan(1, Mnum);
MSSIM   = nan(1, Mnum);
MFSIM   = nan(1, Mnum);
ERGAS   = nan(1, Mnum);
MSAM    = nan(1, Mnum);

%% ======================== 6. Noisy input ====================
Results{1} = Nhsi;
Time(1) = 0;
Iter(1) = 0;
[MPSNR(1), MSSIM(1), MFSIM(1), ERGAS(1), MSAM(1)] = ...
    HSI_QA(Ohsi * 255, Results{1} * 255);

%% ==================== 7. Reduced options ====================
baseReduced = struct();
baseReduced.r          = r3_common;
baseReduced.maxIter    = maxIter_common;
baseReduced.rho        = rho_common;
baseReduced.tol        = tol_common;
baseReduced.lambda     = lambda_reduced;
baseReduced.DEBUG      = DEBUG;
baseReduced.Slice_weights = slice_weights_reduced;

%% -------- Variant 1: Full AC^2-TRPCA --------
idx = 2;
if Run_Full_AC2
    fprintf('\n[%d/%d] Running %s ...\n', idx-1, Mnum-1, methodName{idx});
    opts = baseReduced;
    opts.prox_P = P_common;
    opts.prox_w = w_ccsvs;
    opts.learn_D = true;

    tic;
    [Xhat, ~, Iter(idx)] = reduced_sep_ablation(Nhsi, opts);
    Time(idx) = toc;
    Results{idx} = postprocess_result(Xhat, clip_output);
    [MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
        HSI_QA(Ohsi * 255, Results{idx} * 255);
end

%% -------- Variant 2: Remove concave compensation --------
idx = 3;
if Run_No_Concave
    fprintf('\n[%d/%d] Running %s ...\n', idx-1, Mnum-1, methodName{idx});
    opts = baseReduced;
    opts.prox_P = 0;
    opts.prox_w = w_no_concave;
    opts.learn_D = true;

    tic;
    [Xhat, ~, Iter(idx)] = reduced_sep_ablation(Nhsi, opts);
    Time(idx) = toc;
    Results{idx} = postprocess_result(Xhat, clip_output);
    [MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
        HSI_QA(Ohsi * 255, Results{idx} * 255);
end

%% -------- Variant 3: Fixed transform D --------
idx = 4;
if Run_Fixed_D
    fprintf('\n[%d/%d] Running %s ...\n', idx-1, Mnum-1, methodName{idx});
    opts = baseReduced;
    opts.prox_P = P_common;
    opts.prox_w = w_ccsvs;
    opts.learn_D = false;  % freeze the initial V_hat, i.e. transform D

    tic;
    [Xhat, ~, Iter(idx)] = reduced_sep_ablation(Nhsi, opts);
    Time(idx) = toc;
    Results{idx} = postprocess_result(Xhat, clip_output);
    [MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
        HSI_QA(Ohsi * 255, Results{idx} * 255);
end

%% -------- Variant 4: Learned D + standard TNN (ATNN) --------
idx = 5;
if Run_ATNN
    fprintf('\n[%d/%d] Running %s ...\n', idx-1, Mnum-1, methodName{idx});
    opts = struct();
    opts.r       = r3_common;
    opts.maxIter = maxIter_common;
    opts.rho     = rho_common;
    opts.tol     = tol_common;
    opts.lambda  = lambda_reduced;
    opts.DEBUG   = DEBUG;

    tic;
    [Xhat, ~, Iter(idx)] = atnn_unified_stop(Nhsi, opts);
    Time(idx) = toc;
    Results{idx} = postprocess_result(Xhat, clip_output);
    [MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
        HSI_QA(Ohsi * 255, Results{idx} * 255);
end

%% -------- Variant 5: Full-size CCSVS --------
idx = 6;
if Run_Full_CCSVS
    fprintf('\n[%d/%d] Running %s ...\n', idx-1, Mnum-1, methodName{idx});
    opts = struct();
    opts.tol           = tol_common;
    opts.max_iter      = maxIter_common;
    opts.rho           = rho_common;
    opts.mu            = mu_full;
    opts.max_mu        = max_mu_full;
    opts.DEBUG         = DEBUG;
    opts.w             = w_ccsvs;
    opts.P             = P_common;
    opts.Slice_weights = slice_weights_full;

    tic;
    [Xhat, ~, ~, ~, Iter(idx)] = full_ccsvs_unified_stop(Nhsi, lambda_full, opts);
    Time(idx) = toc;
    Results{idx} = postprocess_result(Xhat, clip_output);
    [MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
        HSI_QA(Ohsi * 255, Results{idx} * 255);
end

%% ======================= 8. Results table ===================
AblationTable = table(methodName', MPSNR', MSSIM', MFSIM', ERGAS', MSAM', Time', Iter', ...
    'VariableNames', {'Method','MPSNR','MSSIM','MFSIM','ERGAS','MSAM','Time_s','Iterations'});

disp(' ');
disp('================ AC^2-TRPCA Ablation Results ================');
disp(AblationTable);

%% ======================== 9. Save results ===================
noiseTag = make_noise_tag(noise_mode, val_sparse, val_gauss);
saveRoad = fullfile(rootDir, 'results', 'AC2_Ablation', dataName, noiseTag);
if ~exist(saveRoad, 'dir')
    mkdir(saveRoad);
end

settings = struct();
settings.dataName = dataName;
settings.dataSize = [height, width, band];
settings.noise_mode = noise_mode;
settings.val_sparse = val_sparse;
settings.val_gauss = val_gauss;
settings.noise_seed = noise_seed;
settings.r3_common = r3_common;
settings.P_common = P_common;
settings.tol_common = tol_common;
settings.maxIter_common = maxIter_common;
settings.rho_common = rho_common;
settings.lambda_reduced = lambda_reduced;
settings.lambda_full = lambda_full;
settings.w_ccsvs = w_ccsvs;
settings.slice_weights_reduced = slice_weights_reduced;
settings.slice_weights_full = slice_weights_full;
settings.clip_output = clip_output;

save(fullfile(saveRoad, 'Ablation_Results.mat'), ...
    'Results', 'AblationTable', 'settings', 'Ohsi', 'Nhsi');
writetable(AblationTable, fullfile(saveRoad, 'Ablation_Results.csv'));

% Save one representative band/frame and a pseudo-color image when possible.
selected_band = min(25, band);
imageDir = fullfile(saveRoad, 'Images');
if ~exist(imageDir, 'dir')
    mkdir(imageDir);
end
for k = 1:Mnum
    if isempty(Results{k})
        continue;
    end
    imwrite(mat2gray(real(Results{k}(:,:,selected_band))), ...
        fullfile(imageDir, sprintf('%02d_%s_band_%d.png', k, sanitize_name(methodName{k}), selected_band)));
end

if band >= 3 && exist('PseudoImage', 'file') == 2
    rgbBands = choose_rgb_bands(band);
    for k = 1:Mnum
        if isempty(Results{k})
            continue;
        end
        rgb = PseudoImage(real(Results{k}), rgbBands);
        imwrite(min(max(rgb,0),1), ...
            fullfile(imageDir, sprintf('%02d_%s_pseudo.png', k, sanitize_name(methodName{k}))));
    end
end

fprintf('\nFinished. Results saved to:\n%s\n', saveRoad);

%% ========================================================================
%% Local function 1: reduced CCSVS, with learned or fixed transform D
%% ========================================================================
function [X, E_tensor, iter] = reduced_sep_ablation(Ten, opts)
[m,n,p] = size(Ten);

r          = opts.r;
maxIter    = opts.maxIter;
rho        = opts.rho;
tol        = opts.tol;
lambda     = opts.lambda;
w          = opts.prox_w;
P          = opts.prox_P;
Slice_weights = opts.Slice_weights;
learn_D    = opts.learn_D;
DEBUG      = get_opt(opts, 'DEBUG', 0);

if numel(Slice_weights) ~= r
    error('Reduced Slice_weights must have length r3=%d.', r);
end

% Mode-3 matrix representation. In the manuscript, the transform D is
% represented by V_hat in this implementation.
Data = reshape(Ten, [m*n, p]);
Y = Data;
[u0, s0, v0] = svd(Data, 'econ');
if r > size(v0,2)
    error('r3=%d exceeds the available third-mode rank %d.', r, size(v0,2));
end
U_hat = u0(:,1:r) * s0(1:r,1:r);
V_hat = v0(:,1:r);

norm_two = largest_singular_value(Y);
norm_inf = norm(Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / max(dual_norm, eps);

A_hat = U_hat * V_hat';
E_hat = zeros(m*n, p);
mu = 1 / max(norm_two, eps);
mu_bar = mu * 1e7;
data_norm = max(norm(Data, 'fro'), eps);

converged = false;
for iter = 1:maxIter
    % Sparse component update.
    temp = Data - A_hat + Y/mu;
    E_hat = sign(temp) .* max(abs(temp) - lambda/mu, 0);

    % Reduced core update.
    tmp = Data - E_hat + Y/mu;
    tmpU = tmp * V_hat;
    N_tensor = reshape(tmpU, [m, n, r]);
    [B_tensor, tnn] = prox_tnnsep(N_tensor, 1/mu, w, P, Slice_weights);
    U_hat = reshape(B_tensor, [m*n, r]);

    % Adaptive transform update. For the Fixed-D ablation, V_hat remains
    % equal to its common SVD initialization throughout all iterations.
    if learn_D
        [Qu,~,Qv] = svd(tmp' * U_hat, 'econ');
        V_hat = Qu * Qv';
    end

    A_hat = U_hat * V_hat';

    % Multiplier and penalty updates.
    residual = Data - A_hat - E_hat;
    Y = Y + mu * residual;
    mu = min(mu * rho, mu_bar);

    relRes = norm(residual, 'fro') / data_norm;
    if DEBUG && (iter == 1 || mod(iter, 20) == 0 || relRes < tol)
        obj = tnn + lambda * norm(E_hat(:), 1);
        fprintf('  iter=%4d, relRes=%.3e, obj=%.6e\n', iter, relRes, obj);
    end

    if relRes < tol
        converged = true;
        break;
    end
end

if ~converged && DEBUG
    fprintf('  Maximum iterations reached (%d).\n', maxIter);
end

X = reshape(A_hat, [m,n,p]);
E_tensor = reshape(E_hat, [m,n,p]);
end

%% ========================================================================
%% Local function 2: ATNN with the same relative-residual stopping rule
%% ========================================================================
function [X, E_tensor, iter] = atnn_unified_stop(Ten, opts)
[m,n,p] = size(Ten);
r       = opts.r;
maxIter = opts.maxIter;
rho     = opts.rho;
tol     = opts.tol;
lambda  = opts.lambda;
DEBUG   = get_opt(opts, 'DEBUG', 0);

Data = reshape(Ten, [m*n,p]);
Y = Data;
[u0,s0,v0] = svd(Data,'econ');
if r > size(v0,2)
    error('r3=%d exceeds the available third-mode rank %d.', r, size(v0,2));
end
U_hat = u0(:,1:r) * s0(1:r,1:r);
V_hat = v0(:,1:r);

norm_two = largest_singular_value(Y);
norm_inf = norm(Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / max(dual_norm, eps);

A_hat = U_hat * V_hat';
E_hat = zeros(m*n,p);
mu = 1 / max(norm_two, eps);
mu_bar = mu * 1e7;
data_norm = max(norm(Data,'fro'), eps);

converged = false;
for iter = 1:maxIter
    temp = Data - A_hat + Y/mu;
    E_hat = sign(temp) .* max(abs(temp) - lambda/mu, 0);

    tmp = Data - E_hat + Y/mu;
    tmpU = tmp * V_hat;
    tnnL = 0;

    for i = 1:r
        [Us,Ss,Vs] = svd(reshape(tmpU(:,i), [m,n]), 'econ');
        sigma = diag(Ss);
        keep = find(sigma > 1/mu);
        if isempty(keep)
            U_hat(:,i) = 0;
        else
            q = keep(end);
            shrunk = sigma(1:q) - 1/mu;
            coreSlice = Us(:,1:q) * diag(shrunk) * Vs(:,1:q)';
            U_hat(:,i) = coreSlice(:);
            tnnL = tnnL + sum(shrunk);
        end
    end

    [Qu,~,Qv] = svd(tmp' * U_hat, 'econ');
    V_hat = Qu * Qv';
    A_hat = U_hat * V_hat';

    residual = Data - A_hat - E_hat;
    Y = Y + mu * residual;
    mu = min(mu * rho, mu_bar);

    relRes = norm(residual,'fro') / data_norm;
    if DEBUG && (iter == 1 || mod(iter,20) == 0 || relRes < tol)
        obj = tnnL + lambda * norm(E_hat(:),1);
        fprintf('  iter=%4d, relRes=%.3e, obj=%.6e\n', iter, relRes, obj);
    end

    if relRes < tol
        converged = true;
        break;
    end
end

if ~converged && DEBUG
    fprintf('  Maximum iterations reached (%d).\n', maxIter);
end

X = reshape(A_hat, [m,n,p]);
E_tensor = reshape(E_hat, [m,n,p]);
end

%% ========================================================================
%% Local function 3: full-size CCSVS with the same stopping rule
%% ========================================================================
function [L,S,obj,err,iter] = full_ccsvs_unified_stop(X, lambda, opts)
tol       = opts.tol;
max_iter  = opts.max_iter;
rho       = opts.rho;
mu        = opts.mu;
max_mu    = opts.max_mu;
DEBUG     = get_opt(opts, 'DEBUG', 0);
w         = opts.w;
P         = opts.P;
Slice_weights = opts.Slice_weights;

if numel(Slice_weights) ~= size(X,3)
    error('Full-size Slice_weights must have length n3=%d.', size(X,3));
end

L = zeros(size(X));
S = zeros(size(X));
Y = zeros(size(X));
data_norm = max(norm(X(:),'fro'), eps);
converged = false;

for iter = 1:max_iter
    [L,tnnL] = prox_tnn(-S + X - Y/mu, 1/mu, w, P, Slice_weights);
    S = prox_l1(-L + X - Y/mu, lambda/mu);

    residual = L + S - X;
    relRes = norm(residual(:),'fro') / data_norm;

    if DEBUG && (iter == 1 || mod(iter,20) == 0 || relRes < tol)
        obj = tnnL + lambda * norm(S(:),1);
        fprintf('  iter=%4d, relRes=%.3e, obj=%.6e\n', iter, relRes, obj);
    end

    if relRes < tol
        converged = true;
        break;
    end

    Y = Y + mu * residual;
    mu = min(rho * mu, max_mu);
end

if ~converged && DEBUG
    fprintf('  Maximum iterations reached (%d).\n', max_iter);
end

obj = tnnL + lambda * norm(S(:),1);
err = norm(residual(:),'fro');
end

%% ============================= Utilities ================================
function value = get_opt(opts, fieldName, defaultValue)
if isfield(opts, fieldName)
    value = opts.(fieldName);
else
    value = defaultValue;
end
end

function smax = largest_singular_value(A)
try
    smax = lansvd(A, 1, 'L');
catch
    try
        smax = svds(A, 1);
    catch
        s = svd(A, 'econ');
        smax = s(1);
    end
end
end

function X = extract_3d_tensor(S)
if isfield(S, 'Ori_H')
    X = S.Ori_H;
    return;
end
names = fieldnames(S);
for i = 1:numel(names)
    candidate = S.(names{i});
    if isnumeric(candidate) && ndims(candidate) == 3
        X = candidate;
        fprintf('Using tensor variable "%s" from MAT file.\n', names{i});
        return;
    end
end
error('No numeric third-order tensor was found in the MAT file.');
end

function noisy = add_test_noise(clean, mode, sparseLevel, gaussianLevel)
[~,~,n3] = size(clean);
noisy = zeros(size(clean));
for k = 1:n3
    band = clean(:,:,k);
    switch mode
        case 1
            noisy(:,:,k) = imnoise(band, 'salt & pepper', sparseLevel);
        case 2
            noisy(:,:,k) = band + gaussianLevel * randn(size(band));
        case 3
            temp = band + gaussianLevel * randn(size(band));
            noisy(:,:,k) = imnoise(temp, 'salt & pepper', sparseLevel);
        otherwise
            error('noise_mode must be 1, 2, or 3.');
    end
end
end

function X = postprocess_result(X, clipOutput)
X = real(X);
if clipOutput
    X = min(max(X,0),1);
end
end

function tag = make_noise_tag(mode, sparseLevel, gaussianLevel)
switch mode
    case 1
        tag = sprintf('s_%g', sparseLevel);
    case 2
        tag = sprintf('g_%g', gaussianLevel);
    case 3
        tag = sprintf('g_%g_s_%g', gaussianLevel, sparseLevel);
end
tag = strrep(tag, '.', 'p');
end

function bands = choose_rgb_bands(n3)
preferred = [49,27,7];
if all(preferred <= n3)
    bands = preferred;
else
    bands = unique(max(1, round([0.8,0.5,0.2] * n3)), 'stable');
    if numel(bands) < 3
        bands = [n3, max(1,round(n3/2)), 1];
    end
end
end

function name = sanitize_name(name)
name = regexprep(name, '[^a-zA-Z0-9_-]', '_');
end
