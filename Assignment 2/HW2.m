%% given code
% figure(1)
% [y, Fs] = audioread('GNR.m4a');
% tr_gnr = length(y)/Fs; % record time in seconds
% plot((1:length(y))/Fs,y);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('Sweet Child O Mine');
% p8 = audioplayer(y,Fs); playblocking(p8);

%% music score for GNR
clear all; close all; clc;
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
L = tr_gnr;
n = length(y);
t1 = linspace(0,L,n+1);
t = t1(1:n);
k = (2*pi/L)*[0:n/2-1, -n/2:-1];
ks = fftshift(k);

% Gabor filter
step = 100;
t_g = linspace(0, t(end), step);
spec = zeros(length(t_g), n);

for i=1:length(t_g)
    gabor = exp(-100 * (t - t_g(i)).^2);
    gt = fft(gabor .* y');
    gts = abs(fftshift(gt));
    [val, ind] = max(gts(n/2:end));
    [a,b] = ind2sub(size(gts),ind+n/2-1);
    guassian = exp(-0.001 * (ks - ks(b)).^2);
    spec(i,:) = abs(fftshift(gt).*guassian);
end

% spectrogram
figure(1)
pcolor(t_g, ks/(2*pi), log(spec'+1)), shading interp
colormap('hot'), xlabel('Time (s)'), ylabel('Frequency (Hz)')
axis([0,tr_gnr,0,1500])
title('Spectrogram of GNR')

%% music score for Floyd
clear all; close all; clc;
[y, Fs] = audioread('Floyd.m4a');
y = y(1:end-1, 1); % reject the last data to avoid error
tr_floyd = length(y)/Fs; % record time in seconds
L = tr_floyd; 
n = length(y);
t1 = linspace(0,L,n+1);
t = t1(1:n);
k = (2*pi/L)*[0:n/2-1, -n/2:-1];
ks = fftshift(k);

% Gabor filter
step = 100;
t_g = linspace(0, t(end), step);
spec = zeros(length(t_g), n);

for i=1:length(t_g)
    gabor = exp(-100 * (t - t_g(i)).^2);
    gt = fft(gabor .* y');
    gts = abs(fftshift(gt));
    [val, ind] = max(gts(n/2:end));
    [a,b] = ind2sub(size(gts),ind+n/2-1);
    guassian = exp(-0.001 * (ks - ks(b)).^2);
    spec(i,:) = abs(fftshift(gt).*guassian);
end

% spectrogram
figure(2)
pcolor(t_g, ks/(2*pi), log(spec'+1)), shading interp
colormap('hot'), xlabel('Time (s)'), ylabel('Frequency (Hz)')
axis([0,tr_floyd,0,500])
title('Spectrogram of Floyd')

%% filter in frequency domain = 250Hz
clear all; close all; clc;
[y, Fs] = audioread('Floyd.m4a');
itv = 10; % interval = 10 sec per plot

figure(3)
for i=1:length(y)/(Fs*itv)+1
    if i*itv*Fs < length(y) % load part
        [y_s, Fs] = audioread('Floyd.m4a', [(i-1)*itv*Fs+1, i*itv*Fs]);
    else
        [y_s, Fs] = audioread('Floyd.m4a', [(i-1)*itv*Fs+1, length(y)-1]);
    end
    
    tr_floyd = length(y_s)/Fs; % record time in seconds
    L = tr_floyd;
    n = length(y_s);
    t1 = linspace(0, L, n + 1);
    t = t1(1:n);
    k = (2*pi/L)*[0:n/2-1, -n/2:-1];
    ks = fftshift(k);

    % Gabor filter
    step = 100;
    t_g = linspace(0, t(end), step);
    spec = zeros(length(t_g), n);

    for j=1:length(t_g)
        gabor = exp(-100 * (t - t_g(j)).^2);
        gt = fft(gabor.*y_s');
        gts = abs(fftshift(gt));
        [val, ind] = max(gts(n/2:n/2+2*pi*250)); % frequency = 250
        [a,b] = ind2sub(size(gts),ind+n/2-1);
        guassian = exp(-0.001 * (ks - ks(b)).^2);
        spec(j,:) = abs(fftshift(gt).*guassian);
    end
    
    % spectrogram
    subplot(2,3,i)
    pcolor(t_g+itv*(i-1), ks/(2*pi), log(spec'+1)), shading interp
    colormap('hot'), xlabel('Time (s)'), ylabel('Frequency (Hz)')
    axis([itv*(i-1), tr_floyd+itv*(i-1), 0, 300])
    drawnow
end
sgtitle('Spectrogram of Floyd with Filter in Frequency Domain')

%% guitar solo
clear all; close all; clc;
[y, Fs] = audioread('Floyd.m4a');
itv = 10; % interval = 10 sec per plot

figure(4)
for i=1:length(y)/(Fs*itv)+1
    if i*itv*Fs < length(y) % load part
        [y_s, Fs] = audioread('Floyd.m4a', [(i-1)*itv*Fs+1, i*itv*Fs]);
    else
        [y_s, Fs] = audioread('Floyd.m4a', [(i-1)*itv*Fs+1, length(y)-1]);
    end
    
    tr_floyd = length(y_s)/Fs; % record time in seconds
    L = tr_floyd;
    n = length(y_s);
    t1 = linspace(0, L, n + 1);
    t = t1(1:n);
    k = (2*pi/L)*[0:n/2-1, -n/2:-1];
    ks = fftshift(k);

    % Gabor filter
    step = 100;
    t_g = linspace(0, t(end), step);
    spec = zeros(length(t_g), n);

    % filter out bass
    for j=1:length(t_g) 
        gabor = exp(-100 * (t - t_g(j)).^2);
        gt = fft(gabor .* y_s');
        gts = abs(fftshift(gt));
        [val, ind] = max(gts(n/2:n/2+2*pi*250)); % freq of bass
        [a,b] = ind2sub(size(gts),ind+n/2-1);
        guassian = exp(-0.001 * (ks - ks(b)).^2);
        gts = abs(fftshift(gt).*(1-guassian));

        % filter for the guitar
        [val, ind] = max(gts(n/2:n/2+2*pi*1200));
        [a,b] = ind2sub(size(gts), ind+n/2-1);
        guassian = exp(-0.001 * (ks - ks(b)).^2);
        spec(j,:) = abs(fftshift(gt).*guassian);
    end
    
    % spectrogram
    subplot(2,3,i)
    pcolor(t_g+itv*(i-1), ks/(2*pi), log(spec'+1)), shading interp
    colormap('hot'), xlabel('Time (s)'), ylabel('Frequency (Hz)')
    axis([itv*(i-1), tr_floyd+itv*(i-1), 0, 1200])
    drawnow
end
sgtitle('Spectrogram of Floyd (Guitar)')