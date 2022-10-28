clear;clc;
% Input parameters

% Make sure your input image is well centered (i.e. the MI slice is not
% touching the boundary and there is enough gap)

% use overlap_image() to see the overlap after warping.

% change parameters for threshold_Histogram() depending on the noise in your dataset. Read the paper for more details. 

% The code is modular. Hence comment/switch a module if its not required.

% The code is well commented for ease of use/modification. Also use
% the aux folders for debugging.

% The final non-linear transfromation uses Laplace Equations, can be easily
% replaced to cubic B-spline depending on the applicatin. 

channelInfo=2;     % channel used for registration (v.imp)
alpha=50;          % medial_axis length (edge_length)
thresh1=0;
R=0;               % no rotation              

if(isempty(gcp('nocreate'))==1)
    parpool;                                        
end                             ?
warning('off','all')

%% ---------------------------Reading data from file--------------------------
base_dir=pwd;                    
addpath(genpath(base_dir))      

mkdir(fullfile(base_dir,'registered_images'));          % aux folders to store data to debug
mkdir(fullfile(base_dir,'info'));
mkdir(fullfile(base_dir,'overlap_images'));
mkdir(fullfile(base_dir,'dump(edge_images)'));              
wdir=fullfile(base_dir,'registered_images');             % for writing the registered images
txt_dir=(fullfile(base_dir,'info'));                     % for writing the data into text file
overlapdir=fullfile(base_dir,'overlap_images');          % for storing the overlaped images
edge_dir = fullfile(base_dir,'dump(edge_images)');       % for storing the edge images-for testing histogram threshold

rdir=(fullfile(base_dir,'data'));                                   
a=fullfile(rdir,'/img*tif');                                                   
dirinfo=dir(fullfile(rdir,'/nolabeldelet750size'));               % sorting the MI images
size0=size(dirinfo);
length=size0(1);
filename=str2mat(dirinfo.name);
for i=1:length-2
    name1{i}=dirinfo(i+2).name;
    new_name1{i}=name1{i}(1:end-4);
end
a=str2double(new_name1);
sortedImages=sort(a);                                %
len=size(sortedImages);
size1=len(2);

for i=1:length
     name1{i}=dirinfo(i).name;
 end
 sortedImages1=sort(name1);     


%% Registration 

% for sliceNum=1913:length
for sliceNum=1:length
tic ;%tic2
disp(['Matching ',num2str(sortedImages(sliceNum)),' microscopic slice to atlas slice']); 
idex=num2str(sortedImages(sliceNum));
idex_suffix=[idex,'.jpg'];
idir=fullfile(rdir,'/nolabeldelet750size/');

atlas=imread(fullfile(idir,idex_suffix));
image=select_AI(sortedImages(sliceNum));
%atlas=imread(fullfile(rdir,'atlas',sortedAtlas{sliceNum+2}));
edgepath=fullfile(edge_dir,sprintf('%04d',sliceNum));
mkdir(edgepath);

I=image;
edge_atlas=atlas_segmentation(atlas); 
[xatlas_normal,yatlas_normal]=normal_vector(edge_atlas,false);
atlasorientation=atand(yatlas_normal./xatlas_normal);   


%------------------------Rotation Correction----------------------------
%  [warped_image,thresh1,R] = rotation_alignment(I,edge_atlas,channelInfo,edgepath);
% figure('Name','AFTER_ROTATION'),imshow(warped_image)

%------------------------Bounding Box Alignment(Scaling & Translation)-----
[warped_image,thresh2,T] = bb_alignment(I,edge_atlas,channelInfo,edgepath);
% figure('Name','AFTER_BOUNDINGBOX'),imshow(warped_image)
 
%---------------------------ICP registration--------------------
[warped_image,num_of_Corres,transf_Matrices,thresh3] = icp_registration(warped_image,edge_atlas,atlasorientation,channelInfo,edgepath,sliceNum);
imwrite(warped_image+(255-atlas),fullfile(edgepath,'4_After_all_warping.jpg')); 

close all
pause(1);

% imwrite(output_image,fullfile(wdir,sortedImages1{sliceNum+2}),'tif');
 imwrite(output_image,fullfile(wdir,idex_suffix),'tif');
t=rgb2gray(atlas);
t=edge(t,'canny');
t=largestConnectedComponent(t,500,false);
t=uint8(t*255);
overlap_atlas=cat(3,t,t,t);
%imwrite((output_image+overlap_atlas),fullfile(overlapdir,sortedImages1{sliceNum+2}),'tif');
imwrite((output_image+overlap_atlas),fullfile(overlapdir,idex_suffix),'tif');

%---------------------------Writing_Data_to_Files--------------------------
save(fullfile(edgepath,'imageData.mat'),'warped_image','output_image','atlas'); 
%ind_txtdir=fullfile(txt_dir,regexprep(sortedImages1{sliceNum+2},'.tif','.txt')); 
%ind_txtdir=fullfile(txt_dir,regexprep(idex_suffix,'.tif','.txt')); 
ind_txtdir=fullfile(txt_dir,regexprep(idex_suffix,'.jpg','.txt')); 
fileID = fopen(ind_txtdir,'w');
fprintf(fileID,'\nThresholds:\n');
fprintf(fileID,'thresh1: %f\t thresh2: %f\t thresh3: %f\t thresh4: %f\t\n',...
    thresh1,thresh2,thresh3);
% fprintf(fileID,'\n#Correspondences: %f %f %f %f #finalCorres %f\n',num_of_Corres(1),num_of_Corres(2),...
%     num_of_Corres(3),num_of_Corres(4),finalCorresNum);
fprintf(fileID,'\nRotation Angle: %f\n',R);
fprintf(fileID,'\nInitial Bounding_Box_Transformation:\n');
dlmwrite(ind_txtdir,T,'-append','delimiter','\t','precision','%.4f');
fclose(fileID);

fileID = fopen(ind_txtdir,'a');
fprintf(fileID,'\nICP_transformations:\n\n');
dlmwrite(ind_txtdir,transf_Matrices{1},'-append','delimiter','\t','precision','%.4f');
fprintf(fileID,'\n');
dlmwrite(ind_txtdir,transf_Matrices{2},'-append','delimiter','\t','precision','%.4f');
fprintf(fileID,'\n');
dlmwrite(ind_txtdir,transf_Matrices{3},'-append','delimiter','\t','precision','%.4f');
fprintf(fileID,'\n');
dlmwrite(ind_txtdir,transf_Matrices{4},'-append','delimiter','\t','precision','%.4f');
fclose(fileID);
end
 disp(['toc is compute the time in ',num2str(sliceNum),'is:',num2str(toc)]);


