%% Given code
% Clean workspace
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); x = x2(1:n); y = x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% for j=1:49
%     Un(:,:,:)=reshape(subdata(:,j),n,n,n);
%     M = max(abs(Un),[],'all');
%     close all, isosurface(X,Y,Z,abs(Un)/M,0.7)
%     axis([-20 20 -20 20 -20 20]), grid on, drawnow
%     pause(1)
% end

%% Part 1 Center Frequency
% average data
ave = zeros(n,n,n);
for j = 1:49
    utnave(:,:,:) = reshape(subdata(:,j),n,n,n);
    ave = ave + utnave;
end
ave = abs(fftn(ave))/49;

% center frequency
[utnmax,ind] = max(ave(:));
[cx,cy,cz] = ind2sub([64,64,64],ind);
centerFreq = [Kx(cx,cy,cz) Ky(cx,cy,cz) Kz(cx,cy,cz)];

%% Part 2 Plot Trajectory
% Gaussian filter
tau = 0.2;
filter = exp(-tau.*((Kx-centerFreq(1)).^2 + (Ky-centerFreq(2)).^2 + (Kz-centerFreq(3)).^2));

% get positions
for i = 1:49
    utn = fftn(reshape(subdata(:,i),n,n,n));
    unf = ifftn(filter .* utn);
    [unfmax,ind] = max(abs(unf(:)));
    [x,y,z] = ind2sub([64,64,64],ind);
    xpos(i) = X(x,y,z);
    ypos(i) = Y(x,y,z);
    zpos(i) = Z(x,y,z);
end

% plot trajectory
plot3(xpos,ypos,zpos)
set(gca,'Fontsize',12)
xlabel('X')
ylabel('Y')
zlabel('Z')
title('Trajectory of the Submarine Over Time','Fontsize',18)
grid on, hold on

% start point
plot3(xpos(1),ypos(1),zpos(1),'r.','Markersize',18)
% end point
plot3(xpos(49),ypos(49),zpos(49),'k.','Markersize',18)
legend('Trajectory','Start Point','End Point')

%% Part 3 Table of x- and y-position
Table = table(xpos', ypos', 'VariableNames', {'x', 'y'});