function [svm_results] = SVMclassifier_salinas_le(img,train_samples,train_labels)


[r, c, b] = size(img);
img = reshape(img,r*c,b);
% Normalize the training set and original image
[train_samples,M,m] = scale_func(train_samples);
[img] = scale_func(img,M,m);

% Select the paramter for SVM with five-fold cross validation
[Ccv Gcv cv cv_t] = cross_validation_svm(train_labels,train_samples);

% Training using a Gaussian RBF kernel
%give the parameters of the SVM (Thanks Pedram for providing the
% parameters of the SVM)
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv); 

%%% Train the SVM
model=svmtrain(train_labels,train_samples,parameter);

[svm_results] = svmpredict(ones(r*c,1),img,model);  