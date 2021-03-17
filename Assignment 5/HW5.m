%% Monte carlo
clear all; close all; clc;
monte = VideoReader('monte_carlo_low.mp4');

dt = 1/monte.Framerate;
t = 0:dt:monte.Duration;
video = zeros(monte.Width*monte.Height, monte.NumberOfFrames);
for i=1:monte.NumberOfFrames
    frame = rgb2gray(read(monte,i));
    video(:,i)=double(reshape(frame,monte.Width*monte.Height,1));
end

% DMD
X1 = video(:,1:end-1);
X2 = video(:,2:end);
[u,s,v] = svd(X1,'econ');
lamda = diag(s);

plot(lamda/sum(lamda),'.','Markersize',20);
xlabel('Singular Value'),ylabel('Proportion');
title('Singular value spectrum');
saveas(gcf,'monte spec.png');

r = 25;
u_r = u(:,1:r);
s_r = s(1:r,1:r);
v_r = v(:,1:r);
Atilde = u_r'*X2*v_r/s_r;
[W,D] = eig(Atilde); 
Phi = X2*v_r/s_r*W;
mu = diag(D);
omega = log(mu)/dt;

% low-rank and sparse
y0 = Phi\X1(:,1);
X_modes = zeros(length(y0),length(t)-1);
for iter = 1:(length(t)-1)
    X_modes(:,iter) = y0.*exp(omega*t(iter));
end
X_dmd = Phi*X_modes;

X_s = X1-abs(X_dmd);
R = X_s.*(X_s<0);
X_sparse = X_s-R;
X_sparse_2 = X_sparse+200;

% picture
original = reshape(uint8(video(:,200)), monte.Height, monte.Width);
background = reshape(uint8(X_dmd(:,200)), monte.Height, monte.Width);
foreground = reshape(uint8(X_sparse(:,200)), monte.Height, monte.Width);
foreground2 = reshape(uint8(X_sparse_2(:,200)), monte.Height, monte.Width);

figure()
subplot(2,2,1);
imshow(original);
title('Original Video');

subplot(2,2,2);
imshow(background);
title('Background');

subplot(2,2,3);
imshow(foreground);
title('Foreground');

subplot(2,2,4);
imshow(foreground2);
title('Foreground (Brighter)');
saveas(gcf,'monte.png');

%% Ski drop
clear all; close all; clc;
ski = VideoReader('ski_drop_low.mp4');

dt = 1/ski.Framerate;
t = 0:dt:ski.Duration;
video = zeros(ski.Width*ski.Height, ski.NumberOfFrames);
for i=1:ski.NumberOfFrames
    frame = rgb2gray(read(ski,i));
    video(:,i)=double(reshape(frame,ski.Width*ski.Height,1));
end

% DMD
X1 = video(:,1:end-1);
X2 = video(:,2:end);
[u,s,v] = svd(X1,'econ');
lamda = diag(s);

plot(lamda/sum(lamda),'.','Markersize',20);
xlabel('Singular Value'),ylabel('Proportion');
title('Singular value spectrum');
saveas(gcf,'ski spec.png');

r = 10;
u_r = u(:,1:r);
s_r = s(1:r,1:r);
v_r = v(:,1:r);
Atilde = u_r'*X2*v_r/s_r;
[W,D] = eig(Atilde); 
Phi = X2*v_r/s_r*W;
mu = diag(D);
omega = log(mu)/dt;

% low-rank and sparse
y0 = Phi\X1(:,1);
X_modes = zeros(length(y0),length(t)-1);
for iter = 1:(length(t)-1)
    X_modes(:,iter) = y0.*exp(omega*t(iter));
end
X_dmd = Phi*X_modes;

X_s = X1-abs(X_dmd);
R = X_s.*(X_s<0);
X_sparse = X_s-R;
X_sparse_2 = X_sparse+200;

% picture
original = reshape(uint8(video(:,200)), ski.Height, ski.Width);
background = reshape(uint8(X_dmd(:,200)), ski.Height, ski.Width);
foreground = reshape(uint8(X_sparse(:,200)), ski.Height, ski.Width);
foreground2 = reshape(uint8(X_sparse_2(:,200)), ski.Height, ski.Width);

figure()
subplot(2,2,1);
imshow(original);
title('Original Video');

subplot(2,2,2);
imshow(background);
title('Background');

subplot(2,2,3);
imshow(foreground);
title('Foreground');

subplot(2,2,4);
imshow(foreground2);
title('Foreground (Brighter)');
saveas(gcf,'ski.png');