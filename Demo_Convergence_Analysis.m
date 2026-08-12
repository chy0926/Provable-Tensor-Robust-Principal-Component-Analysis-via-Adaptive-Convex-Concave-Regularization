%% Demo: Convergence Analysis 
clc; clear; close all;

rootDir = fileparts(mfilename('fullpath'));
cd(rootDir);

addpath(genpath(fullfile(rootDir, 'lib')));
addpath(genpath(fullfile(rootDir, 'data')));
addpath(genpath(fullfile(rootDir, 'utils')));
addpath(genpath(fullfile(rootDir, 'RPCA_Code')));

dataName = 'pure_DCmall';
if exist(fullfile('data',[dataName,'.mat']),'file')
    load(fullfile('data',dataName));
else
    error('数据文件未找到');
end

if exist('Ohsi','var')
elseif exist('Ori_H','var')
    Ohsi = Ori_H;
else
    error('未找到图像变量');
end

fprintf('原始数据类型: %s, 值域: [%.4f, %.4f]\n', ...
    class(Ohsi), double(min(Ohsi(:))), double(max(Ohsi(:))));

if isa(Ohsi, 'uint8')
    Ohsi = double(Ohsi) / 255.0;
    fprintf('已将 uint8 转换为 double [0,1]\n');
elseif isa(Ohsi, 'double') && max(Ohsi(:)) > 1
    Ohsi = Ohsi / 255.0;
    fprintf('已将 double [0,255] 归一化为 [0,1]\n');
else
    fprintf('数据已在 [0,1]，无需处理\n');
end

[height, width, band] = size(Ohsi);

rng(1997);
noise_S = 0.1;
Nhsi = zeros(size(Ohsi));
for ii = 1:band
    Nhsi(:,:,ii) = imnoise(Ohsi(:,:,ii), 'salt & pepper', noise_S);
end
fprintf('Sparse noise level: %.1f\n', noise_S);

fprintf('Nhsi 值域: [%.4f, %.4f]\n', min(Nhsi(:)), max(Nhsi(:)));
init_err = norm(reshape(Nhsi,[],band) - reshape(Ohsi,[],band),'fro') / ...
           norm(reshape(Ohsi,[],band),'fro');
fprintf('加噪后相对误差（期望约等于noise_S）: %.4f\n', init_err);

opts            = [];
opts.maxIter    = 100;
opts.tol        = 1e-6;
opts.initial_mu = 5e-2;
opts.rho        = 1.1;
weight_rpca     = 0.5;
opts.lambda     = weight_rpca / sqrt(max(height, width));
opts.r          = 8;
P               = 45;
opts.prox_w     = [linspace(1e-5, 1e-7, P)'; ...
                   0.7 * ones(min(height,width) - P, 1)];
opts.prox_P          = P;
opts.Slice_weights   = [1, ones(1, opts.r-1)];

fprintf('Running AC^2-TRPCA...\n');
[X_rec, E_rec, total_iter, errHistory, timeHistory] = ...
    ATNN_RPCAsep_HOU_Record(Nhsi, opts, Ohsi);
fprintf('Finished! Total iter = %d, Final error = %.4e\n', ...
    total_iter, errHistory(end));

figure('Color','w','Position',[400,300,650,480]);
iters = 1:length(errHistory);
ax = gca;

yyaxis left
plot(iters, errHistory, 'r-', 'LineWidth', 1.5);
ylabel('Relative Error', 'FontSize', 13)
ylim([0, max(errHistory) * 1.05])
ax.YAxis(1).Color = 'r';

yyaxis right
plot(iters, timeHistory, 'b--', 'LineWidth', 1.5);
ylabel('Time (seconds)', 'FontSize', 13)
ylim([0, max(timeHistory) * 1.05])
ax.YAxis(2).Color = 'b';

xlabel('Iterations', 'FontSize', 13)
nTicks = 5;
step = max(1, floor(length(iters) / nTicks));
xticks(1:step:length(iters))
xlim([1, length(iters)])

legend({'Relative Error','Time'}, 'Location','northeast','FontSize',12)