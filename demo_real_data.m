%% Demo: HSI TRPCA Denoising (Auto Path: s / g / g_s)
clc;
clear; close all;

%% ========== 强制固定工作目录 ==========
rootDir = fileparts(mfilename('fullpath'));
cd(rootDir);
rng('default'); rng(1997);

%% Add paths
addpath(genpath(fullfile(rootDir, 'lib')));
addpath(genpath(fullfile(rootDir, 'data')));
addpath(genpath(fullfile(rootDir, 'utils')));

%% ========== 设置其他方法路径 ==========
ATNNSEP_path = fullfile(rootDir, 'RPCA_Code');
CCSVS_path   = fullfile(rootDir, 'RPCA_Code', 'TRPCA', 'CCSVS');
ATNN_path    = fullfile(rootDir, 'RPCA_Code');
CTV_path     = fullfile(rootDir, 'RPCA_Code');
E3DTV_path   = fullfile(rootDir, 'RPCA_Code', 'TRPCA', 'E3DTV');
RCTV_path    = fullfile(rootDir, 'RPCA_Code', 'TRPCA');

%% Data settings
dataName = 'Lobby'; %hsi_PaC;Curtain;pure_DCmall;PaviaU
dataRoad = fullfile(rootDir, 'data', dataName); 
load(dataRoad);
Ohsi = Ori_H;%data;Ori_H
[height, width, band] = size(Ohsi);
dim = [height, width, band];

%% Enable bits
Run_RPCA      = 0;
Run_SNN       = 0;
Run_KBR       = 0;
Run_TCTV      = 0;
Run_E3DTV     = 0;
Run_LRTV      = 0;
Run_RCTV      = 0;
Run_WRCTV     = 0;
Run_CCSVS     = 0;
Run_ATNN      = 0;
Run_CTV       = 0;
Run_ATNNSEP   = 1; %This is the method of this paper.
%% Save options
getGIF         = 1;
getBand        = 1;
selected_band  = 25;
getPseudoImage = 1;
selected_bands = [49, 27, 7];

%% Method list
methodName = {'Noisy', 'RPCA', 'SNN', 'KBR', 'TCTV', 'E3DTV', 'LRTV', 'RCTV', 'WRCTV', 'ATNNSEP', 'CCSVS', 'ATNN', 'CTV'};
Mnum = length(methodName);

%% =============================================================
%% ===== Noise Settings & Path Generation =====
%% =============================================================
noise_mode = 3;

val_sparse = 0.1;
val_gauss  = 0.1;

folder_suffix = '';

switch noise_mode
    case 1
        sparselevel = val_sparse;
        sparsesigma = sparselevel * ones(band, 1);
        folder_suffix = ['s', erase(num2str(sparselevel), '.')];
        disp(['=== Mode: Sparse Only | Level: ', num2str(sparselevel), ' ===']);
        
    case 2
        noiselevel = val_gauss;
        gausssigma = noiselevel * ones(band, 1);
        folder_suffix = ['g', erase(num2str(noiselevel), '.')];
        disp(['=== Mode: Gaussian Only | Level: ', num2str(noiselevel), ' ===']);
        
    case 3
        noiselevel  = val_gauss; 
        sparselevel = val_sparse; 
        gausssigma  = noiselevel * ones(band, 1);
        sparsesigma = sparselevel * ones(band, 1);
        folder_suffix = ['g_s', erase(num2str(sparselevel), '.')]; 
        disp(['=== Mode: Mixed (G+S) | Gauss: ', num2str(noiselevel), ' + Sparse: ', num2str(sparselevel), ' ===']);
end

rootDir = fileparts(mfilename('fullpath'));
saveRoad = fullfile(rootDir, 'results', 'CHEN_TRPCA', ['results_for_', dataName], folder_suffix);

fprintf('=== 结果将保存在: %s\n', saveRoad);

if ~exist(saveRoad,'dir'), mkdir(saveRoad); end
gifPath = fullfile(saveRoad,'GIF');
bandPath = fullfile(saveRoad,'Band');
pseudoPath = fullfile(saveRoad,'PseudoImage');
if getGIF && ~exist(gifPath,'dir'); mkdir(gifPath); end
if getBand && ~exist(bandPath,'dir'); mkdir(bandPath); end
if getPseudoImage && ~exist(pseudoPath,'dir'); mkdir(pseudoPath); end

%% Load previous results
if exist(fullfile(saveRoad,'QA_Results.mat'),'file')
    load(fullfile(saveRoad,'QA_Results.mat'));
end
if exist(fullfile(saveRoad,'Results.mat'),'file')
    load(fullfile(saveRoad,'Results.mat'));
end

%% Save Original
if getGIF; togetGif(Ohsi, fullfile(gifPath,'Ohsi')); end
if getBand; imwrite(Ohsi(:,:,selected_band), fullfile(bandPath,'Ohsi.jpg')); end
if getPseudoImage; imwrite(PseudoImage(Ohsi, selected_bands), fullfile(pseudoPath,'Ohsi.jpg')); end

%% =============================================================
%% ===== Add Noise =====
%% =============================================================
rng(42);
Nhsi = zeros(size(Ohsi));

disp('正在添加噪声...');
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

%% Allocate results
Results = cell(1, Mnum);
Time = zeros(1, Mnum);
MPSNR = zeros(1, Mnum);
MSSIM = zeros(1, Mnum);
MFSIM = zeros(1, Mnum);
ERGAS = zeros(1, Mnum);
MSAM = zeros(1, Mnum);

%% Noisy QA
i = 1;
Results{i} = Nhsi;
[MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
Time(i) = 0;
if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
enList = 1;

%% ================== Run RPCA ==================
i = i+1;
if Run_RPCA
    addpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    disp(['Running ',methodName{i}, ' ... ']);
    D = zeros(height*width, band);
    for j=1:band
        bandp = Nhsi(:,:,j);
        D(:,j)= bandp(:);
    end
    tic;
    A_hat = rpca_m(D);
    Results{i} = reshape(A_hat, [height, width, band]);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    enList = [enList, i];
end

%% ================== Run SNN ==================
i = i+1;
if Run_SNN
    addpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    disp(['Running ',methodName{i}, ' ... ']);
    opts = [];
    opts.DEBUG = 1;
    tic;
    alpha = [1, 1, 200];
    Results{i} = trpca_snn(Nhsi, alpha, opts);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    enList = [enList, i];
end

%% ================== Run KBR ==================
i = i+1;
if Run_KBR
    addpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    disp(['Running ',methodName{i}, ' ... ']);
    beta           = 2.5*sqrt(max(dim));
    gamma          = beta*100;
    Par.maxIter    = 1000;
    Par.lambda     = 0.1;
    Par.mu         = 10;
    Par.tol        = 1e-5;
    Par.rho        = 1.1;
    tic;
    Results{i} = KBR_RPCA(Nhsi, beta, gamma, Par);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    enList = [enList, i];
end

%% ================== Run TCTV ==================
i = i+1;
if Run_TCTV
    addpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    disp(['Running ',methodName{i}, ' ... ']);
    opts = [];
    opts.rho = 1.25;
    opts.directions = [1,2,3];
    tic;
    Results{i} = TCTV_TRPCA(Nhsi, opts);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    enList = [enList, i];
end

%% ================== Run E3DTV ==================
i = i+1;
if Run_E3DTV
    addpath(genpath(E3DTV_path));
    disp(['Running ',methodName{i}, ' ... ']);
    r = [4, 4, 4];
    tau = 0.004 * sqrt(height * width);
    tic;
    Results{i} = EnhancedTV(Nhsi, tau, r);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(E3DTV_path));
    enList = [enList, i];
end

%% ================== Run LRTV ==================
i = i+1;
if Run_LRTV
    addpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    disp(['Running ',methodName{i}, ' ... ']);
    tau = 0.005;
    lambda = 5/sqrt(height*width);
    rank = 13;
    tic;
    Results{i} = LRTV_accelerate(Nhsi, tau, lambda, rank);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    enList = [enList, i];
end

%% ================== Run RCTV ==================
i = i+1;
if Run_RCTV
    addpath(genpath(RCTV_path));
    disp(['Running ',methodName{i}, ' ... ']);
    r = 13;
    beta = 50;
    lambda = 1;
    tau = [0.8, 0.8];
    tic;
    Results{i} = RCTV(Nhsi, beta, lambda, tau, r);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(RCTV_path));
    enList = [enList, i];
end

%% ================== Run WRCTV ==================
i = i+1;
if Run_WRCTV
    addpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    disp(['Running ',methodName{i}, ' ... ']);
    r = 13;
    beta = 50;
    lambda = 0.30;
    tau = [0.8, 0.8];
    rho = 1.05;
    tic;
    Results{i} = RCTV_IRNN(Nhsi, beta, lambda, tau, r, rho);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(fullfile(rootDir, 'RPCA_Code', 'TRPCA', methodName{i})));
    enList = [enList, i];
end

%% ================== Run ATNNSEP ==================
i = i+1;
if Run_ATNNSEP
    addpath(genpath(ATNNSEP_path));
    disp(['Running ',methodName{i}, ' ... ']);
    weight_rpca = 1;
    rk = 3;
    opts = [];
    opts.DEBUG = 1;
    opts.lambda = weight_rpca / sqrt(max(height, width));
    opts.r = rk;
    P = 0;
    opts.prox_w = [linspace(1e-5, 1e-7, P)'; 0.7*ones(min(height, width)-P, 1)];
    opts.prox_P = P;
   opts.Slice_weights = [1, 3*ones(1, opts.r-1)];
    tic;
    Results{i} = ATNN_RPCAsep_HOU(Nhsi, opts);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(ATNNSEP_path));
    enList = [enList, i];
end

%% ================== Run CCSVS ==================
i = i+1;
if Run_CCSVS
    addpath(genpath(CCSVS_path));
    disp(['Running ',methodName{i}, ' ... ']);
    opts = [];
    opts.mu = 1e-2;
    opts.tol = 1e-5;
    opts.rho = 1.1;
    opts.max_iter = 500;
    opts.DEBUG = 1;
    ll = min(height, width);
    P = 1;
    opts.w = [linspace(1e-5, 1e-7, P)'; 0.7*ones(ll-P, 1)];
    opts.P = P;
    opts.Slice_weights = [0.55, 1.55*ones(1, band-1)];
    lambda = 1/ sqrt(max(height, width) * band);
    tic;
    [L, ~, ~, ~, ~] = trpca_tnn(Nhsi, lambda, opts);
    Results{i} = max(L, 0);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(CCSVS_path));
    enList = [enList, i];
end

%% ================== Run ATNN ==================
i = i+1;
if Run_ATNN
    addpath(genpath(ATNN_path));
    disp(['Running ',methodName{i}, ' ... ']);
    weight_rpca =1;
    rk = 16;
    opts = [];
    opts.DEBUG = 1;
    opts.lambda = weight_rpca / sqrt(max(height, width));
    opts.r = rk;
    tic;
    Results{i} = ATNN_RPCA(Nhsi, opts);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(ATNN_path));
    enList = [enList, i];
end

%% ================== Run CTV ==================
i = i+1;
if Run_CTV
    addpath(genpath(CTV_path));
    disp(['Running ',methodName{i}, ' ... ']);
    weight_rpca = 1;
    opts = [];
    opts.lambda = 3 * weight_rpca / sqrt(height * width);
    opts.weight = 1;
    tic;
    Results{i} = CTV_RPCA(Nhsi, opts);
    Time(i) = toc;
    [MPSNR(i), MSSIM(i), MFSIM(i), ERGAS(i), MSAM(i)] = HSI_QA(Ohsi * 255, Results{i} * 255);
    if getGIF; togetGif(Results{i}, fullfile(gifPath, methodName{i})); end
    if getBand; imwrite(Results{i}(:,:,selected_band), fullfile(bandPath, [methodName{i}, '.jpg'])); end
    if getPseudoImage; imwrite(PseudoImage(Results{i}, selected_bands), fullfile(pseudoPath, [methodName{i}, '.jpg'])); end
    rmpath(genpath(CTV_path));
    enList = [enList, i];
end

%% ================== Show QA Table ==================
fprintf('\n');
fprintf('================== QA Results =====================\n');
fprintf(' %10s    %7s    %7s    %7s    %7s    %7s    %7s\n',...
    'Method', 'MPSNR', 'MSSIM', 'MFSIM', 'ERGAS', 'MSAM', 'Time');
for k = 1:length(enList)
    idx = enList(k);
    fprintf(' %10s   %7.4f   %7.4f   %7.4f   %7.4f   %7.4f   %7.4f\n',...
        methodName{idx}, MPSNR(idx), MSSIM(idx), MFSIM(idx), ERGAS(idx), MSAM(idx), Time(idx));
end
fprintf('================== Show Results =====================\n');
close all;
showHSIResult(Results, Ohsi, methodName, enList, selected_band, band);

%% ================== Save results ==================
All = [MPSNR; MSSIM; MFSIM; ERGAS; MSAM; Time];
save(fullfile(saveRoad,'QAResults.mat'), 'All', 'MPSNR', 'MSSIM', 'MFSIM', 'ERGAS', 'MSAM', 'Time');
save(fullfile(saveRoad,'Results.mat'), 'Results');
disp(['✅ 任务完成! 结果保存在: ' saveRoad]);