clear all; close all; clc;
[images_train, labels_train] = mnist_parse('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[images_test, labels_test] = mnist_parse('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');

%% SVD
data_train = reshape(images_train, [784 60000]);
data_test = reshape(images_test, [784 10000]);

[u,s,v] = svd(double(data_train),'econ');
train = (s*v')';
test = (double(u)'*double(data_test))';
lambda = diag(s);

%% Singular value spectrum
figure()
plot(lambda/sum(lambda),'.','Markersize',20)
xlabel('Singular Value'),ylabel('Proportion')
title('Singular value spectrum')
saveas(gcf,'spectrum.png')

%% V-modes
figure()
colormap jet
for i = 0:9
    ind = find(labels_train == i);
    scatter3(v(ind,2),v(ind,3),v(ind,5),20,labels_train(ind),'.')
    hold on
end
xlabel('Column 2')
ylabel('Column 3')
zlabel('Column 5')
legend({'0','1','2','3','4','5','6','7','8','9'});
title('3D Projection of V-modes')
saveas(gcf,'vmode.png')

%% LDA for two digits
LDA_acc = zeros(10,10);
for i = 0:8
    for j = i+1:9
        x1_train = train(labels_train == i,2:10);
        x2_train = train(labels_train == j,2:10);
        xtrain = [x1_train; x2_train];
        x1_len = size(x1_train,1);
        x2_len = size(x2_train,1);
        numtrain = [i*ones(x1_len,1); j*ones(x2_len,1)];
        
        x1_test = test(labels_test == i,2:10);
        x2_test = test(labels_test == j,2:10);
        xtest = [x1_test; x2_test];
        x1_len = size(x1_test,1);
        x2_len = size(x2_test,1);
        numtest = [i*ones(x1_len,1); j*ones(x2_len,1)];
        prediction = classify(xtest,xtrain,numtrain);
        error = sum(numtest-prediction ~= 0);
        LDA_acc(i+1,j+1) = 1-error/length(numtest);
    end
end

%% LDA for 0,1,2
x1_train = train(labels_train == 0,2:10);
x2_train = train(labels_train == 1,2:10);
x3_train = train(labels_train == 2,2:10);
xtrain = [x1_train; x2_train; x3_train];
x1_len = size(x1_train,1);
x2_len = size(x2_train,1);
x3_len = size(x3_train,1);
numtrain = [0*ones(x1_len,1); 1*ones(x2_len,1); 2*ones(x3_len,1)];

x1_test = test(labels_test == 0,2:10);
x2_test = test(labels_test == 1,2:10);
x3_test = test(labels_test == 2,2:10);
xtest = [x1_test; x2_test; x3_test];
x1_len = size(x1_test,1);
x2_len = size(x2_test,1);
x3_len = size(x3_test,1);
numtest = [0*ones(x1_len,1); 1*ones(x2_len,1); 2*ones(x3_len,1)];
prediction = classify(xtest,xtrain,numtrain);
error = sum(numtest-prediction ~= 0);
acc = 1-error/length(numtest);

%% LDA for all
xtrain = train(:,2:10);
xtest = test(:,2:10);
prediction = classify(xtest,xtrain,labels_train);
error = sum(labels_test-prediction ~= 0);
LDA_acc_all = 1-error/length(labels_test);

%% SVM for all
xtrain = train(:,2:10)/max(max(s));
xtest = test(:,2:10)/max(max(s));

SVMModel = cell(10,1);
classes = 0:9;
rng(1);
for i = 1:numel(classes)
    ind = labels_train == classes(i);
    SVMModel{i} = fitcsvm(xtrain,ind,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
end
for i = 1:numel(classes)
    [~,score] = predict(SVMModel{i},xtest);
    Pos_score(:,i) = score(:,2);
end

[~,max] = max(Pos_score,[],2);
error = sum(labels_test+1-max ~= 0);
SVM_acc = 1-error/length(labels_test);

%% Decision Tree for all
xtrain = train(:,2:10);
xtest = test(:,2:10);
Mdl = fitctree(xtrain,labels_train,'OptimizeHyperparameters','auto');
prediction = predict(Mdl,xtest);
error = sum(labels_test-prediction ~= 0);
DT_acc = 1-error/length(labels_test);

%% Compare performace on pairs
% easiest (0,1)
x1_train = train(labels_train == 0,2:10);
x2_train = train(labels_train == 1,2:10);
x1_len = size(x1_train,1);
x2_len = size(x2_train,1);
xtrain = [x1_train; x2_train];
numtrain = [0*ones(x1_len,1); 1*ones(x2_len,1)];

x1_test = test(labels_test == 0,2:10);
x2_test = test(labels_test == 1,2:10);
xtest = [x1_test; x2_test];
x1_len = size(x1_test,1);
x2_len = size(x2_test,1);
numtest = [0*ones(x1_len,1); 1*ones(x2_len,1)];

% SVM
rng default
Mdl_SVM = fitcsvm(xtrain,numtrain,'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
prediction = predict(Mdl_SVM,xtest);
error = sum(numtest-prediction ~= 0);
SVM_acc_e = 1-error/length(numtest);

% Decision tree
Mdl_DT = fitctree(xtrain,numtrain,'OptimizeHyperparameters','auto');
prediction = predict(Mdl_DT,xtest);
error = sum(numtest-prediction ~= 0);
DT_acc_e = 1-error/length(numtest);

% hardest (4,9)
x1_train = train(labels_train == 4,2:10);
x2_train = train(labels_train == 9,2:10);
xtrain = [x1_train; x2_train];
x1_len = size(x1_train,1);
x2_len = size(x2_train,1);
numtrain = [4*ones(x1_len,1); 9*ones(x2_len,1)];

x1_test = test(labels_test == 4,2:10);
x2_test = test(labels_test == 9,2:10);
xtest = [x1_test; x2_test];
x1_len = size(x1_test,1);
x2_len = size(x2_test,1);
numtest = [4*ones(x1_len,1); 9*ones(x2_len,1)];

% SVM
rng default
Mdl_SVM = fitcsvm(xtrain,numtrain,'Standardize',true,...
        'KernelFunction','rbf','BoxConstraint',1);
prediction = predict(Mdl_SVM,xtest);
error = sum(numtest-prediction ~= 0);
SVM_acc_h = 1-error/length(numtest);

% Decision tree
Mdl_DT = fitctree(xtrain,numtrain,'OptimizeHyperparameters','auto');
prediction = predict(Mdl_DT,xtest);
error = sum(numtest-prediction ~= 0);
DT_acc_h = 1-error/length(numtest);