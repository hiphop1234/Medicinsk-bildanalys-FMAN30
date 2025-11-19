%% Code for part 1 of shape model project 
clc
clear
close all

% Load the DMSA images and manual segmentation
load dmsa_images
load man_seg

% Extract x- and y-coordinates
Xmcoord=real(man_seg);
Ymcoord=imag(man_seg);

% Choose patient and look at image
pat_nbr = 1;

figure
imagesc(dmsa_images(:,:,pat_nbr))
colormap(gray)
axis xy
axis equal
hold on
drawshape_comp(man_seg,[1 length(man_seg) 1],'.-r')


%% Code for part 2 of shape model project 
clc
clear
close all

% Load the DMSA images
load dmsa_images

% Choose patient and look at image
pat_nbr = 1;

figure
imagesc(dmsa_images(:,:,pat_nbr))
colormap(gray)
axis xy

% Load the manual segmentations
% Columns 1-20: the right kidney of patient 1-20
% Columns 21-40: the mirrored left kidney of patient 1-20
% Each row is a landmark position
load models

% Extract x- and y-coordinates
Xcoord=real(models);
Ycoord=imag(models);

% Mirror the left kidney to get it in the right position in the image
figure
imagesc(dmsa_images(:,:,pat_nbr))
colormap(gray)
axis xy
hold on
drawshape_comp(models(:,pat_nbr),[1 14 1],'.-r')
drawshape_comp((size(dmsa_images,2)+1)-models(:,pat_nbr+20)',[1 14 1],'.-r')
axis equal



















