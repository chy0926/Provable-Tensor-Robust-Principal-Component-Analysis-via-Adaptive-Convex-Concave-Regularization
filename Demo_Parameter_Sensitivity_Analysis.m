%% ========================================================================
% Demo: Parameter Sensitivity Analysis for ATNNSEP (AC^2-TRPCA)
% Data & Noise: Strictly follows your main script (Curtain, Mixed Noise)
% Parameters to analyze:
%   1. Lambda (Regularization Parameter)
%   2. P      (Separation Parameter)
%   3. r      (Dimension Reduction Parameter / r3)
% ========================================================================

clc;
clear; close all;

cd(fileparts(mfilename('fullpath')));
rng('default'); rng(1997);

addpath(genpath('lib'));
addpath(genpath('data'));
addpath(genpath('TC_Code'));
addpath(genpath('utils'));

ATNNSEP_path = 'D:\adaptive_tensor_nuclear_norm-main';
addpath(genpath(ATNNSEP_path));

dataName = 'Lobby'; 
dataRoad = fullfile('data', dataName); 
load(dataRoad);
Ohsi = Ori_H;
[height, width, band] = size(Ohsi);
dim = [height, width, band];

noise_mode = 3;  

val_sparse = 0.1;  
val_gauss  = 0.1;  

gausssigma  = val_gauss * ones(band, 1);
sparsesigma = val_sparse * ones(band, 1);

rng(42);
Nhsi = zeros(size(Ohsi));
fprintf('正在生成带噪数据 (Mode=%d, Gauss=%.2f, Sparse=%.2f)...\n', noise_mode, val_gauss, val_sparse);

for ii = 1:band
    currentBand = Ohsi(:,:,ii);
    switch noise_mode
        case 1
            Nhsi(:,:,ii) = imnoise(currentBand, 'salt & pepper', sparsesigma(ii));
        case 2
            Nhsi(:,:,ii) = currentBand + gausssigma(ii) * randn(height, width);
        case 3
            temp = currentBand + gausssigma(ii) * randn(height, width);
            Nhsi(:,:,ii) = imnoise(temp, 'salt & pepper', sparsesigma(ii));
    end
end

base_weight_rpca = 1;
base_P = 3;
base_r = 4;

lambda_vals = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0];
P_vals = [1, 2, 3, 4, 5, 6, 8, 10];
r_vals = [2, 3, 4, 5, 8, 10, 15, 20]; 

Res_Lambda = struct('x', lambda_vals, 'PSNR', [], 'Time', []);
Res_P      = struct('x', P_vals,      'PSNR', [], 'Time', []);
Res_r      = struct('x', r_vals,      'PSNR', [], 'Time', []);

saveDir = fullfile('results', 'Sensitivity_Analysis', dataName);
if ~exist(saveDir, 'dir'), mkdir(saveDir); end

fprintf('\n=== Experiment 1: Sensitivity to Lambda (weight_rpca) ===\n');
psnr_list = zeros(length(lambda_vals), 1);
time_list = zeros(length(lambda_vals), 1);

for i = 1:length(lambda_vals)
    w_rpca = lambda_vals(i);
    
    opts = [];
    opts.DEBUG = 0;
    
    opts.lambda = w_rpca / sqrt(max(height, width));
    
    opts.r = base_r;
    opts.prox_P = base_P;
    opts.prox_w = [linspace(1e-5, 1e-7, base_P)'; 1.5*ones(min(height, width)-base_P, 1)];
    opts.Slice_weights = [1, 1*ones(1, opts.r-1)];
    
    tic;
    [X_hat, ~, ~] = ATNN_RPCAsep_HOU(Nhsi, opts);
    t_cost = toc;
    
    [mpsnr, ~, ~, ~, ~] = HSI_QA(Ohsi * 255, X_hat * 255);
    psnr_list(i) = mpsnr;
    time_list(i) = t_cost;
    
    fprintf('Weight: %.1f | PSNR: %.4f | Time: %.2f s\n', w_rpca, mpsnr, t_cost);
end
Res_Lambda.PSNR = psnr_list;
Res_Lambda.Time = time_list;

fprintf('\n=== Experiment 2: Sensitivity to Parameter P ===\n');
psnr_list = zeros(length(P_vals), 1);
time_list = zeros(length(P_vals), 1);

for i = 1:length(P_vals)
    p_curr = P_vals(i);
    
    opts = [];
    opts.DEBUG = 0;
    
    opts.prox_P = p_curr;
    
    ll = min(height, width);
    p_safe = min(p_curr, ll);
    opts.prox_w = [linspace(1e-5, 1e-7, p_safe)'; 1.5*ones(ll-p_safe, 1)];
    
    opts.lambda = base_weight_rpca / sqrt(max(height, width));
    opts.r = base_r;
    opts.Slice_weights = [1, 1*ones(1, opts.r-1)];
    
    tic;
    [X_hat, ~, ~] = ATNN_RPCAsep_HOU(Nhsi, opts);
    t_cost = toc;
    
    [mpsnr, ~, ~, ~, ~] = HSI_QA(Ohsi * 255, X_hat * 255);
    psnr_list(i) = mpsnr;
    time_list(i) = t_cost;
    
    fprintf('P: %d | PSNR: %.4f | Time: %.2f s\n', p_curr, mpsnr, t_cost);
end
Res_P.PSNR = psnr_list;
Res_P.Time = time_list;

fprintf('\n=== Experiment 3: Sensitivity to Dimension r (r3) ===\n');
psnr_list = zeros(length(r_vals), 1);
time_list = zeros(length(r_vals), 1);

for i = 1:length(r_vals)
    r_curr = r_vals(i);
    
    opts = [];
    opts.DEBUG = 0;
    
    opts.r = r_curr;
    
    opts.Slice_weights = [1, 1*ones(1, r_curr-1)];
    
    opts.lambda = base_weight_rpca / sqrt(max(height, width));
    opts.prox_P = base_P;
    opts.prox_w = [linspace(1e-5, 1e-7, base_P)'; 1.5*ones(min(height, width)-base_P, 1)];
    
    tic;
    [X_hat, ~, ~] = ATNN_RPCAsep_HOU(Nhsi, opts);
    t_cost = toc;
    
    [mpsnr, ~, ~, ~, ~] = HSI_QA(Ohsi * 255, X_hat * 255);
    psnr_list(i) = mpsnr;
    time_list(i) = t_cost;
    
    fprintf('r: %d | PSNR: %.4f | Time: %.2f s\n', r_curr, mpsnr, t_cost);
end
Res_r.PSNR = psnr_list;
Res_r.Time = time_list;

save(fullfile(saveDir, 'Sensitivity_Data.mat'), 'Res_Lambda', 'Res_P', 'Res_r');

figure('Units', 'pixels', 'Position', [100, 100, 1200, 350], 'Color', 'w');

subplot(1, 3, 1);
yyaxis left;
plot(Res_Lambda.x, Res_Lambda.PSNR, '-bo', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel('\lambda ');
ylabel('PSNR (dB)');
yyaxis right;
plot(Res_Lambda.x, Res_Lambda.Time, '-r^', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
ylabel('Time (s)');

subplot(1, 3, 2);
yyaxis left;
plot(Res_P.x, Res_P.PSNR, '-bo', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel(' r');
ylabel('PSNR (dB)');
yyaxis right;
plot(Res_P.x, Res_P.Time, '-r^', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
ylabel('Time (s)');

subplot(1, 3, 3);
yyaxis left;
plot(Res_r.x, Res_r.PSNR, '-bo', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
xlabel('r_3');
ylabel('PSNR (dB)');
yyaxis right;
plot(Res_r.x, Res_r.Time, '-r^', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
ylabel('Time (s)');

exportgraphics(gcf, fullfile(saveDir, 'Sensitivity_Lobby.png'), 'Resolution', 300);
fprintf('\nDone! Results saved to %s\n', saveDir);