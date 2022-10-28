function [new_image]=mask_warping(I,atlasnormimage,T)
[r,c]=size(atlasnormimage); % points where we want interpolated values. Has to be atlas points
[xq,yq]=meshgrid(1:c,1:r);
% xq=2045;yq=3000;

[r1,c1,~]=size(I);          % because my microscope image can or cannot be of same size as atlas image
[x,y]=meshgrid(1:c1,1:r1);    
x=reshape(x,[],1);
y=reshape(y,[],1);

% doing the transformation
imagecordold=[x';y';ones(length(x),1)'];
imagecordnew=T*imagecordold;

x=imagecordnew(1,:)';
y=imagecordnew(2,:)';

% tic
% F = scatteredInterpolant(x,y,reshape(double(I(:,:)),[],1),'linear','none');
% new_image(:,:)=F(xq,yq);
% fprintf('Time for image warping is %f secs \n',toc) 

tic
parfor i=1:3
F = scatteredInterpolant(x,y,reshape(double(I(:,:,i)),[],1),'linear','none');
new_image(:,:,i)=F(xq,yq);
end

fprintf('Time for image warping is %f secs \n',toc) 
% figure;imshow(new_image1)
% new_image1=uint8(new_image1);
% figure;imshow(new_image1);

new_image=uint8(new_image);
%imshow(new_image);
end
