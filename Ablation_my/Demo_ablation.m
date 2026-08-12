%% AC^2-TRPCA ablation: original method-specific parameters + unified stopping rule
%   1) Full AC^2-TRPCA
%   2) AC^2-TRPCA without concave compensation
%   3) AC^2-TRPCA with fixed transform D
%   4) Learned D + standard TNN 
%   5) Full-size CCSVS
%

clc;
clear;
close all;

%% ========== Fix working directory and random seed ==========
scriptDir = fileparts(mfilename('fullpath'));
if ~isempty(scriptDir)
    cd(scriptDir);
end
rng('default');
rng(1997);

%% ========== Paths ==========

addpath(genpath('data'));
addpath(genpath('TC_Code'));
addpath(genpath('utils'));


algorithmRoot = 'D:\adaptive_tensor_nuclear_norm-main';
if isfolder(algorithmRoot)
    addpath(genpath(algorithmRoot));
else
    warning('Algorithm folder not found: %s', algorithmRoot);
end

addpath(scriptDir);

requiredFunctions = { ...
    'ATNN_RPCAsep_unifiedstop', ...
    'ATNN_RPCA_unifiedstop', ...
    'trpca_tnn_unifiedstop', ...
    'prox_tnnsep', 'prox_tnn', 'prox_l1', 'HSI_QA'};
for q = 1:numel(requiredFunctions)
    assert(exist(requiredFunctions{q}, 'file') == 2, ...
        'Missing required function: %s', requiredFunctions{q});
end

%% ========== Data settings ==========

dataName = 'PaviaU';
dataRoad = fullfile('data', dataName);
load(dataRoad);

assert(exist('data', 'var') == 1, ...
    'The MAT file must contain the clean tensor variable data.');
Ohsi = double(data);
[height, width, band] = size(Ohsi);
minSpatial = min(height, width);

fprintf('Dataset: %s, size = %d x %d x %d\n', ...
    dataName, height, width, band);

%% ========== Noise settings: identical to the original main script ==========
% 1 = sparse only; 2 = Gaussian only; 3 = Gaussian + sparse
noise_mode = 1;
val_sparse = 0.3;
val_gauss  = 0.1;

switch noise_mode
    case 1
        sparselevel = val_sparse;
        sparsesigma = sparselevel * ones(band, 1);
        folder_suffix = ['s', erase(num2str(sparselevel), '.')];
        fprintf('Noise: sparse only, p = %.4f\n', sparselevel);
    case 2
        noiselevel = val_gauss;
        gausssigma = noiselevel * ones(band, 1);
        folder_suffix = ['g', erase(num2str(noiselevel), '.')];
        fprintf('Noise: Gaussian only, sigma = %.4f\n', noiselevel);
    case 3
        noiselevel  = val_gauss;
        sparselevel = val_sparse;
        gausssigma  = noiselevel * ones(band, 1);
        sparsesigma = sparselevel * ones(band, 1);
        folder_suffix = ['g_s', erase(num2str(sparselevel), '.')];
        fprintf('Noise: Gaussian %.4f + sparse %.4f\n', ...
            noiselevel, sparselevel);
    otherwise
        error('noise_mode must be 1, 2, or 3.');
end


rng(42);
Nhsi = zeros(size(Ohsi));
for ii = 1:band
    currentBand = Ohsi(:, :, ii);
    switch noise_mode
        case 1
            Nhsi(:, :, ii) = imnoise(currentBand, 'salt & pepper', sparsesigma(ii));
        case 2
            Nhsi(:, :, ii) = currentBand + gausssigma(ii) * randn(height, width);
        case 3
            temp = currentBand + gausssigma(ii) * randn(height, width);
            Nhsi(:, :, ii) = imnoise(temp, 'salt & pepper', sparsesigma(ii));
    end
end

fprintf('Clean norm = %.16e; noisy norm = %.16e; noisy sum = %.16e\n', ...
    norm(Ohsi(:)), norm(Nhsi(:)), sum(Nhsi(:)));

%% ========== Unified stopping rule ==========

common_tol     = 1e-5;
common_maxIter = 500;

%% ========== Controlled parameters for reduced-space variants ==========

r3_common = 5;  

lambda_reduced = 1 / sqrt(max(height, width));
rho_reduced    = 1.05;


P_reduced = 0;
base_weight_reduced = 0.7;
top_weights_reduced = linspace(1e-5, 1e-7, P_reduced)';
weights_reduced = [top_weights_reduced; ...
    base_weight_reduced * ones(minSpatial - P_reduced, 1)];
slice_weights_reduced = [1, 1.5* ones(1, r3_common - 1)];

%% ========== Original full-size CCSVS model parameters ==========

ccsvs_mu       = 1e-2;
ccsvs_rho      = 1.1;
ccsvs_max_mu   = 1e10;
ccsvs_P        = min(1, minSpatial);
ccsvs_weights  = [linspace(1e-5, 1e-7, ccsvs_P)'; ...
    0.7 * ones(minSpatial - ccsvs_P, 1)];
ccsvs_slice_weights = [0.55, 1.55 * ones(1, band - 1)];
lambda_full_ccsvs = 1 / sqrt(max(height, width) * band);

%% ========== Output settings ==========
methodName = { ...
    'Noisy', ...
    'AC^2-TRPCA (full model)', ...
    'AC^2-TRPCA w/o concave compensation', ...
    'AC^2-TRPCA with fixed transform D', ...
    'Learned D + standard TNN', ...
    'Full-size CCSVS'};
Mnum = numel(methodName);

Results = cell(1, Mnum);
Time = zeros(1, Mnum);
Iterations = zeros(1, Mnum);
MPSNR = zeros(1, Mnum);
MSSIM = zeros(1, Mnum);
MFSIM = zeros(1, Mnum);
ERGAS = zeros(1, Mnum);
MSAM = zeros(1, Mnum);

rootDir = fileparts(mfilename('fullpath'));
saveRoad = fullfile(rootDir, 'results', 'AC2_Ablation_UnifiedStop', ...
    ['results_for_', dataName], folder_suffix);
if ~exist(saveRoad, 'dir')
    mkdir(saveRoad);
end

%% ========== Noisy ==========
idx = 1;
Results{idx} = Nhsi;
[MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
    HSI_QA(Ohsi * 255, Results{idx} * 255);

%% ========== 1. Full AC2-TRPCA ==========
idx = 2;
opts = struct();
opts.DEBUG = 1;
opts.lambda = lambda_reduced;
opts.r = r3_common;
opts.rho = rho_reduced;
opts.tol = common_tol;
opts.maxIter = common_maxIter;
opts.prox_P = P_reduced;
opts.prox_w = weights_reduced;
opts.Slice_weights = slice_weights_reduced;
opts.learn_D = true;

disp(['Running ', methodName{idx}, ' ...']);
tic;
[Results{idx}, ~, Iterations(idx)] = ATNN_RPCAsep_unifiedstop(Nhsi, opts);
Time(idx) = toc;
[MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
    HSI_QA(Ohsi * 255, Results{idx} * 255);

%% ========== 2. Remove the concave compensation ==========
idx = 3;
opts_wo = opts;
opts_wo.prox_P = 1;
opts_wo.prox_w = base_weight_reduced * ones(minSpatial, 1);
opts_wo.learn_D = true;

disp(['Running ', methodName{idx}, ' ...']);
tic;
[Results{idx}, ~, Iterations(idx)] = ATNN_RPCAsep_unifiedstop(Nhsi, opts_wo);
Time(idx) = toc;
[MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
    HSI_QA(Ohsi * 255, Results{idx} * 255);

%% ========== 3. Fixed transform D ==========

idx = 4;
opts_fixed = opts;
opts_fixed.learn_D = false;

disp(['Running ', methodName{idx}, ' ...']);
tic;
[Results{idx}, ~, Iterations(idx)] = ATNN_RPCAsep_unifiedstop(Nhsi, opts_fixed);
Time(idx) = toc;
[MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
    HSI_QA(Ohsi * 255, Results{idx} * 255);

%% ========== 4. Learned D + standard TNN (ATNN) ==========

idx = 5;
opts_atnn = struct();
opts_atnn.DEBUG = 1;
opts_atnn.lambda = lambda_reduced;
opts_atnn.r = r3_common;
opts_atnn.rho = rho_reduced;
opts_atnn.tol = common_tol;
opts_atnn.maxIter = common_maxIter;

disp(['Running ', methodName{idx}, ' ...']);
tic;
[Results{idx}, ~, Iterations(idx)] = ATNN_RPCA_unifiedstop(Nhsi, opts_atnn);
Time(idx) = toc;
[MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
    HSI_QA(Ohsi * 255, Results{idx} * 255);

%% ========== 5. Full-size CCSVS ==========

idx = 6;
opts_ccsvs = struct();
opts_ccsvs.mu = ccsvs_mu;
opts_ccsvs.rho = ccsvs_rho;
opts_ccsvs.max_mu = ccsvs_max_mu;
opts_ccsvs.tol = common_tol;
opts_ccsvs.max_iter = common_maxIter;
opts_ccsvs.DEBUG = 1;
opts_ccsvs.w = ccsvs_weights;
opts_ccsvs.P = ccsvs_P;
opts_ccsvs.Slice_weights = ccsvs_slice_weights;

disp(['Running ', methodName{idx}, ' ...']);
tic;
[L, ~, ~, ~, Iterations(idx)] = ...
    trpca_tnn_unifiedstop(Nhsi, lambda_full_ccsvs, opts_ccsvs);

Results{idx} = max(L, 0);
Time(idx) = toc;
[MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx)] = ...
    HSI_QA(Ohsi * 255, Results{idx} * 255);

%% ========== Display and save ==========
Method = methodName(:);
ResultTable = table(Method, MPSNR(:), MSSIM(:), MFSIM(:), ERGAS(:), ...
    MSAM(:), Time(:), Iterations(:), ...
    'VariableNames', {'Method', 'MPSNR', 'MSSIM', 'MFSIM', 'ERGAS', ...
    'MSAM', 'Time_s', 'Iterations'});

disp('================ AC^2-TRPCA Ablation Results ================');
disp(ResultTable);

%% ========== Classical ablation figure: accuracy and efficiency ==========

ablationIdx = 2:Mnum;
plotNames = { ...
    'AC^{2}-TRPCA', ...
    'w/o concave compensation', ...
    'fixed transform D', ...
    'learned D + standard TNN', ...
    'full-size CCSVS'};

% Build a concise noise description for the figure title.
switch noise_mode
    case 1
        noiseLabel = sprintf('sparse noise, p = %.1f', val_sparse);
    case 2
        noiseLabel = sprintf('Gaussian noise, \\sigma = %.1f', val_gauss);
    case 3
        noiseLabel = sprintf('mixed noise, G = %.1f, S = %.1f', ...
            val_gauss, val_sparse);
end

figAblation = figure('Color', 'w', ...
    'Name', 'AC2-TRPCA Ablation Study', ...
    'Position', [100, 100, 1250, 470]);
layout = tiledlayout(figAblation, 1, 2, ...
    'TileSpacing', 'compact', 'Padding', 'compact');

% ---------- (a) Reconstruction accuracy ----------
ax1 = nexttile(layout, 1);
psnrVals = MPSNR(ablationIdx);
b1 = bar(ax1, psnrVals, 0.72, 'FaceColor', 'flat', ...
    'EdgeColor', [0.20, 0.20, 0.20], 'LineWidth', 0.8);

b1.CData = repmat([0.72, 0.72, 0.72], numel(psnrVals), 1);
b1.CData(1, :) = [0.20, 0.20, 0.20];
set(ax1, 'XTick', 1:numel(plotNames), ...
    'XTickLabel', plotNames, ...
    'TickLabelInterpreter', 'tex', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 11, 'LineWidth', 1.0);
xtickangle(ax1, 20);
ylabel(ax1, 'MPSNR (dB) \uparrow', 'Interpreter', 'tex');
title(ax1, '(a) Reconstruction accuracy', 'FontWeight', 'normal');
grid(ax1, 'on');
box(ax1, 'on');
ylim(ax1, [0, max(psnrVals) * 1.16]);
for k = 1:numel(psnrVals)
    text(ax1, k, psnrVals(k) + 0.015 * max(psnrVals), ...
        sprintf('%.3f', psnrVals(k)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontName', 'Times New Roman', 'FontSize', 10);
end

% ---------- (b) Computational efficiency ----------
ax2 = nexttile(layout, 2);
timeVals = Time(ablationIdx);
b2 = bar(ax2, timeVals, 0.72, 'FaceColor', 'flat', ...
    'EdgeColor', [0.20, 0.20, 0.20], 'LineWidth', 0.8);
b2.CData = repmat([0.72, 0.72, 0.72], numel(timeVals), 1);
b2.CData(1, :) = [0.20, 0.20, 0.20];
set(ax2, 'XTick', 1:numel(plotNames), ...
    'XTickLabel', plotNames, ...
    'TickLabelInterpreter', 'tex', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 11, 'LineWidth', 1.0);
xtickangle(ax2, 20);
ylabel(ax2, 'Runtime (s) \downarrow', 'Interpreter', 'tex');
title(ax2, '(b) Computational efficiency', 'FontWeight', 'normal');
grid(ax2, 'on');
box(ax2, 'on');
ylim(ax2, [0, max(timeVals) * 1.16]);
for k = 1:numel(timeVals)
    text(ax2, k, timeVals(k) + 0.015 * max(timeVals), ...
        sprintf('%.2f', timeVals(k)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontName', 'Times New Roman', 'FontSize', 10);
end

sgtitle(layout, sprintf('Component-wise Ablation on %s (%s)', ...
    strrep(dataName, '_', '\_'), noiseLabel), ...
    'Interpreter', 'tex', 'FontName', 'Times New Roman', ...
    'FontSize', 13, 'FontWeight', 'normal');


figBase = fullfile(saveRoad, 'Ablation_Accuracy_Runtime');
savefig(figAblation, [figBase, '.fig']);
exportgraphics(figAblation, [figBase, '.png'], 'Resolution', 600);
exportgraphics(figAblation, [figBase, '.pdf'], 'ContentType', 'vector');


%% ========== Multi-metric ablation profile curve ==========

metricNames = {'MPSNR', 'MSSIM', 'MFSIM', 'ERGAS', 'MSAM', 'Runtime'};
profileRaw = [ ...
    MPSNR(ablationIdx).', ...
    MSSIM(ablationIdx).', ...
    MFSIM(ablationIdx).', ...
    ERGAS(ablationIdx).', ...
    MSAM(ablationIdx).', ...
    Time(ablationIdx).'];

profileScore = zeros(size(profileRaw));

% Higher-is-better metrics: MPSNR, MSSIM, and MFSIM.
for c = 1:3
    col = profileRaw(:, c);
    rangeVal = max(col) - min(col);
    if rangeVal < eps
        profileScore(:, c) = 1;
    else
        profileScore(:, c) = (col - min(col)) / rangeVal;
    end
end

% Lower-is-better metrics: ERGAS, MSAM, and runtime.
for c = 4:6
    col = profileRaw(:, c);
    rangeVal = max(col) - min(col);
    if rangeVal < eps
        profileScore(:, c) = 1;
    else
        profileScore(:, c) = (max(col) - col) / rangeVal;
    end
end

figProfile = figure('Color', 'w', ...
    'Name', 'AC2-TRPCA Multi-metric Ablation Profile', ...
    'Position', [120, 120, 1050, 540]);

hold on;
markerSet = {'o', 's', '^', 'd', 'v'};
lineSet = {'-', '--', '-.', ':', '-'};

for m = 1:numel(ablationIdx)
    plot(1:numel(metricNames), profileScore(m, :), ...
        'LineStyle', lineSet{m}, ...
        'Marker', markerSet{m}, ...
        'LineWidth', 1.6, ...
        'MarkerSize', 7);
end
hold off;

set(gca, ...
    'XTick', 1:numel(metricNames), ...
    'XTickLabel', metricNames, ...
    'FontName', 'Times New Roman', ...
    'FontSize', 11, ...
    'LineWidth', 1.0);
xlim([0.75, numel(metricNames) + 0.25]);
ylim([-0.05, 1.05]);
ylabel('Normalized score (higher is better)');
xlabel('Evaluation dimension');
title(sprintf('Normalized Ablation Profile on %s (%s)', ...
    strrep(dataName, '_', '\_'), noiseLabel), ...
    'Interpreter', 'tex', ...
    'FontWeight', 'normal');
grid on;
box on;

legend(plotNames, ...
    'Location', 'southoutside', ...
    'NumColumns', 2, ...
    'Interpreter', 'tex', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 9);

profileBase = fullfile(saveRoad, 'Ablation_Normalized_Profile');
savefig(figProfile, [profileBase, '.fig']);
exportgraphics(figProfile, [profileBase, '.png'], 'Resolution', 600);
exportgraphics(figProfile, [profileBase, '.pdf'], 'ContentType', 'vector');

save(fullfile(saveRoad, 'Ablation_Results.mat'), ...
    'Results', 'ResultTable', 'Ohsi', 'Nhsi', 'methodName', ...
    'common_tol', 'common_maxIter', 'r3_common', 'P_reduced', ...
    'lambda_reduced', 'lambda_full_ccsvs');
writetable(ResultTable, fullfile(saveRoad, 'Ablation_Results.csv'));

fprintf('Results saved to: %s\n', saveRoad);
