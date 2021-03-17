%% Test 1
clear all; close all; clc; 
load('cam1_1.mat');
load('cam2_1.mat');
load('cam3_1.mat');

numFrames11 = size(vidFrames1_1, 4);
numFrames21 = size(vidFrames2_1, 4);
numFrames31 = size(vidFrames3_1, 4);

%% Cam 1
% filter for mass
filter = zeros(480,640);
filter(170:430,300:350) = 1;

% track can
for i=1:numFrames11
    gray = rgb2gray(vidFrames1_1(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x11(i) = mean(X);
    y11(i) = mean(Y);
end

% cut frame
[a,b] = max(y11(1:30));
y11 = y11(b:end);
x11 = x11(b:end);

%% Cam 2
% filter for mass
filter = zeros(480,640);
filter(100:400,230:360) = 1;

% track can
for i=1:numFrames21
    gray = rgb2gray(vidFrames2_1(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x21(i) = mean(X);
    y21(i) = mean(Y);
end

% cut frame
[a,b] = max(y21(1:30));
y21 = y21(b:end);
x21 = x21(b:end);

%% Cam 3
% filter for mass
filter = zeros(480,640);
filter(200:350,240:490) = 1;

% track can
for i=1:numFrames31
    gray = rgb2gray(vidFrames3_1(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x31(i) = mean(X);
    y31(i) = mean(Y);
end

% cut frame
[a,b] = max(y31(1:30));
y31 = y31(b:end);
x31 = x31(b:end);

%% PCA
minlength = min([length(x11),length(x21),length(x31)]);
x11 = x11(1:minlength);
x21 = x21(1:minlength);
x31 = x31(1:minlength);
y11 = y11(1:minlength);
y21 = y21(1:minlength);
y31 = y31(1:minlength);

% svd
X = [x11; y11; x21; y21; x31; y31];
X = X - repmat(mean(X,2),1,minlength);
[u,s,v] = svd(X*X'/sqrt(minlength-1));
lambda = diag(s);

figure()
plot(lambda.^2/sum(lambda.^2), '.','MarkerSize',30)
xlabel('Principal component'),ylabel('Energy')
title('Test 1')
saveas(gcf,'energy1.png')

Y = u'*X;
figure()
plot(Y(1,:),'Linewidth',2), hold on
plot(Y(2,:),'Linewidth',2)
plot(Y(3,:),'Linewidth',2)
xlabel('Frame'),ylabel('Position')
title('Test 1')
legend('PC1', 'PC2', 'PC3')
saveas(gcf,'position1.png')

%% Test 2
clear all; close all; clc; 
load('cam1_2.mat');
load('cam2_2.mat');
load('cam3_2.mat');

numFrames12 = size(vidFrames1_2, 4);
numFrames22 = size(vidFrames2_2, 4);
numFrames32 = size(vidFrames3_2, 4);

%% Cam 1
% filter for mass
filter = zeros(480,640);
filter(170:430,300:400) = 1;

% track can
for i=1:numFrames12
    gray = rgb2gray(vidFrames1_2(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x12(i) = mean(X);
    y12(i) = mean(Y);
end

% cut frame
[a,b] = max(y12(1:30));
y12 = y12(b:end);
x12 = x12(b:end);

%% Cam 2
% filter for mass
filter = zeros(480,640);
filter(50:480,170:420) = 1;

% track can
for i=1:numFrames22
    gray = rgb2gray(vidFrames2_2(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x22(i) = mean(X);
    y22(i) = mean(Y);
end

% cut frame
[a,b] = max(y22(1:30));
y22 = y22(b:end);
x22 = x22(b:end);

%% Cam 3
% filter for mass
filter = zeros(480,640);
filter(200:380,240:490) = 1;

% track can
for i=1:numFrames32
    gray = rgb2gray(vidFrames3_2(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x32(i) = mean(X);
    y32(i) = mean(Y);
end

% cut frame
[a,b] = max(y32(1:30));
y32 = y32(b:end);
x32 = x32(b:end);

%% PCA
minlength = min([length(x12),length(x22),length(x32)]);
x12 = x12(1:minlength);
x22 = x22(1:minlength);
x32 = x32(1:minlength);
y12 = y12(1:minlength);
y22 = y22(1:minlength);
y32 = y32(1:minlength);

% svd
X = [x12; y12; x22; y22; x32; y32];
X = X - repmat(mean(X,2),1,minlength);
[u,s,v] = svd(X*X'/sqrt(minlength-1));
lambda = diag(s);

figure()
plot(lambda.^2/sum(lambda.^2), '.','MarkerSize',30)
xlabel('Principal component'),ylabel('Energy')
title('Test 2')
saveas(gcf,'energy2.png')

Y = u'*X;
figure()
plot(Y(1,:),'Linewidth',2), hold on
plot(Y(2,:),'Linewidth',2)
plot(Y(3,:),'Linewidth',2)
xlabel('Frame'),ylabel('Position')
title('Test 2')
legend('PC1', 'PC2', 'PC3')
saveas(gcf,'position2.png')

%% Test 3
clear all; close all; clc; 
load('cam1_3.mat');
load('cam2_3.mat');
load('cam3_3.mat');

numFrames13 = size(vidFrames1_3, 4);
numFrames23 = size(vidFrames2_3, 4);
numFrames33 = size(vidFrames3_3, 4);

%% Cam 1
% filter for mass
filter = zeros(480,640);
filter(230:450,250:450) = 1;

% track can
for i=1:numFrames13
    gray = rgb2gray(vidFrames1_3(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x13(i) = mean(X);
    y13(i) = mean(Y);
end

% cut frame
[a,b] = max(y13(1:30));
y13 = y13(b:end);
x13 = x13(b:end);

%% Cam 2
% filter for mass
filter = zeros(480,640);
filter(100:420,170:420) = 1;

% track can
for i=1:numFrames23
    gray = rgb2gray(vidFrames2_3(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x23(i) = mean(X);
    y23(i) = mean(Y);
end
 
% cut frame
[a,b] = max(y23(1:30));
y23 = y23(b:end);
x23 = x23(b:end);

%% Cam 3
% filter for mass
filter = zeros(480,640);
filter(160:370,250:600) = 1;

% track can
for i=1:numFrames33
    gray = rgb2gray(vidFrames3_3(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x33(i) = mean(X);
    y33(i) = mean(Y);
end

% cut frame
[a,b] = max(y33(1:30));
y33 = y33(b:end);
x33 = x33(b:end);

%% PCA
minlength = min([length(x13),length(x23),length(x33)]);
x13 = x13(1:minlength);
x23 = x23(1:minlength);
x33 = x33(1:minlength);
y13 = y13(1:minlength);
y23 = y23(1:minlength);
y33 = y33(1:minlength);

% svd
X = [x13; y13; x23; y23; x33; y33];
X = X - repmat(mean(X,2),1,minlength);
[u,s,v] = svd(X*X'/sqrt(minlength-1));
lambda = diag(s);

figure()
plot(lambda.^2/sum(lambda.^2), '.','MarkerSize',30)
xlabel('Principal component'),ylabel('Energy')
title('Test 3')
saveas(gcf,'energy3.png')

Y = u'*X;
figure()
plot(Y(1,:),'Linewidth',2), hold on
plot(Y(2,:),'Linewidth',2)
plot(Y(3,:),'Linewidth',2)
xlabel('Frame'),ylabel('Position')
title('Test 3')
legend('PC1', 'PC2', 'PC3')
saveas(gcf,'position3.png')

%% Test 4
clear all; close all; clc; 
load('cam1_4.mat');
load('cam2_4.mat');
load('cam3_4.mat');

numFrames14 = size(vidFrames1_4, 4);
numFrames24 = size(vidFrames2_4, 4);
numFrames34 = size(vidFrames3_4, 4);

%% Cam 1
% filter for mass
filter = zeros(480,640);
filter(220:450,270:470) = 1;

% track can
for i=1:numFrames14
    gray = rgb2gray(vidFrames1_4(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x14(i) = mean(X);
    y14(i) = mean(Y);
end

% cut frame
[a,b] = max(y14(1:30));
y14 = y14(b:end);
x14 = x14(b:end);

%% Cam 2
% filter for mass
filter = zeros(480,640);
filter(50:400,160:430) = 1;

% track can
for i=1:numFrames24
    gray = rgb2gray(vidFrames2_4(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x24(i) = mean(X);
    y24(i) = mean(Y);
end
 
% cut frame
[a,b] = max(y24(1:30));
y24 = y24(b:end);
x24 = x24(b:end);

%% Cam 3
% filter for mass
filter = zeros(480,640);
filter(100:300,270:500) = 1;

% track can
for i=1:numFrames34
    gray = rgb2gray(vidFrames3_4(:,:,:,i));
    gray_f = double(gray).*filter;
    [a,b] = max(gray_f(:));
    [X,Y] = find(gray_f > a*11/12);
    x34(i) = mean(X);
    y34(i) = mean(Y);
end

% cut frame
[a,b] = max(y34(1:30));
y34 = y34(b:end);
x34 = x34(b:end);

%% PCA
minlength = min([length(x14),length(x24),length(x34)]);
x14 = x14(1:minlength);
x24 = x24(1:minlength);
x34 = x34(1:minlength);
y14 = y14(1:minlength);
y24 = y24(1:minlength);
y34 = y34(1:minlength);

% svd
X = [x14; y14; x24; y24; x34; y34];
X = X - repmat(mean(X,2),1,minlength);
[u,s,v] = svd(X*X'/sqrt(minlength-1));
lambda = diag(s);

figure()
plot(lambda.^2/sum(lambda.^2), '.','MarkerSize',30)
xlabel('Principal component'),ylabel('Energy')
title('Test 4')
saveas(gcf,'energy4.png')

Y = u'*X;
figure()
plot(Y(1,:),'Linewidth',2), hold on
plot(Y(2,:),'Linewidth',2)
plot(Y(3,:),'Linewidth',2)
xlabel('Frame'),ylabel('Position')
title('Test 4')
legend('PC1', 'PC2', 'PC3')
saveas(gcf,'position4.png')