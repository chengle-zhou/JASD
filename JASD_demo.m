% The codes by Chengle Zhou
% Time: 2019.10.27
% density peak based image fusion for HSI

close all; clear all; clc
addpath ('.\common')
% rmpath ('.\libsvm-3.20')
% addpath ('.\libsvm-3.20')
%% load original image
load(['.\datasets\Salinas.mat']);
load(['.\datasets\Salinas_gt.mat']);
img = salinas;
img = img./max(img(:));
no_class = max(salinas_gt(:));

%% record the indexs of background pixels
[bagrX, bagrY] = find(double(salinas_gt) == 0);
[rows, cols, band_ori] = size(img);
img_ori = img;
img = reshape(img, rows * cols, band_ori);
p = 25;

%% PCA reduce dimension
img_pca = compute_mapping(img,'PCA',p);    
edgeI = reshape(img_pca(:,1),rows, cols,1);
edgeI = mat2gray(edgeI);
edgeI = im2uint8(edgeI);

superpixel_data = reshape(img_pca(:,1:3),[rows, cols, 3]);
superpixel_data = mat2gray(superpixel_data);
superpixel_data = im2uint8(superpixel_data);


%% Estimate the superpixel block based on the edge
canny_edge = edge(edgeI,'canny'); % 0.1 
[B,L] = bwboundaries(canny_edge);

lambda_prime = 0.8;  sigma = 10;  conn8 = 1;
number_superpixels = length(B);
SuperLabels = mex_ers(double(superpixel_data),number_superpixels,lambda_prime,sigma,conn8);

%% Generate mean feature map
img_new = reshape(img_pca, rows, cols, p);
[mean_matric,~,~] = mean_feature (img_new,SuperLabels);

%% RF feature construction
RF_feature = spatial_feature(img_new,350,0.4);
fusion_feature = cat(3,mean_matric,RF_feature);
[r,c,b] = size(fusion_feature);
img_fus = fusion_feature;
img_fus_2d = reshape(img_fus,r*c,b);

%% Constructing training and testing data
train_num = [4,8,4,3,5,8,7,20,12,3,3,4,5,4,10,4];      % the rate of training is set to 1.0% of ground thruth
indexes = train_random_select(GroundT(2,:),train_num); % based on 24 for each class
train_SL = GroundT(:,indexes);
test_SL = GroundT;
test_SL(:,indexes) = [];

train_samples = img_fus_2d(train_SL(1,:),:);
train_labels = train_SL(2,:);
test_samples = img_fus_2d(test_SL(1,:),:);
test_labels = test_SL(2,:)';

%% Using SVM classifier to evaluation classification accuracy
[SFE_2_result] = SVMclassifier_salinas_le(img_fus,train_samples,train_labels');

%% Evaluation the performance of the SVM
GroudTest = double(test_labels(:,1));
SFE_2_ResultTest = SFE_2_result(test_SL(1,:),:);
[OA,AA,kappa,CA] = confusion(GroudTest,SFE_2_ResultTest)

