clear;clc;

registrated_file='/mask_save/';
img_name=dir(registrated_file);
size0=size(img_name);
length=size0(1);
for i=1:length-2
    name1{i}=img_name(i+2).name;
    new_name1{i}=name1{i}(1:end-4);
end
a=str2double(name1);
sortedname=sort(a);

for number=5:length-2
    try
    slicenum=sortedname(number);
    fprintf('the idex is %d',slicenum);
    fprintf('the number is %d',number);
    I_rootdir=fullfile('/mask_save/',num2str(slicenum));  %make image
    I_all=dir(I_rootdir);
    length1=size(I_all);
    mask_all=length1(1);
  
    for j=1:length1-2
         tic
        warped_file='/warped_mask/';
        mkdir(fullfile(warped_file,num2str(slicenum)));
        warped_mask=fullfile(warped_file,num2str(slicenum));
        
        add_file='/segmented_regions/';
        mkdir(fullfile(add_file,num2str(slicenum)));
        add_mask=fullfile(add_file,num2str(slicenum));
        
        name2=I_all(j+2).name;  
        new_name2=name2(1:end-4);
        
        I1= imread(fullfile(I_rootdir,name2));  % mask image
        I=imresize(I1,[960,1368]);
        
        atlas1 = imread([fullfile('/IF_images',num2str(slicenum)),'.jpg']); % microimage
        atlas=imresize(atlas1,[1024,1500],'nearest');
        atlasnormimage=atlas_segmentation(atlas);

        filein_file=fullfile('/info/',num2str(slicenum));
        filein=[filein_file,'.txt'];

        % fileout='D:\littleresult\AI_change\36_1.txt';

       %% First_1
%         T1=[0.6991	-0.0000 343.0858;
%           -0.0000	0.7950  -176.9400];
        data10=dataread(filein,10);
        data11=dataread(filein,11);
        T11=str2num(data10);
        T12=str2num(data11);
        T1=[T11;T12];
        new_image1=mask_warping(I,atlasnormimage,T1);
        %figure;imshow(new_image1);
        
        %% Second_2
%         T2=[0.9877	-0.0119	34.3512;
%             -0.0090	 0.9741	16.9589];
        data20=dataread(filein,15);
        data21=dataread(filein,16);
        T21=str2num(data20);
        T22=str2num(data21);
        T2=[T21;T22];
        new_image2=mask_warping(new_image1,atlasnormimage,T2);
        % figure;imshow(new_image2);
       

        %% Third_3
%         T3=[0.9956	0.0011	5.5861;
%             0.0002	0.9980	-8.7342];
        data30=dataread(filein,18);
        data31=dataread(filein,19);
        T31=str2num(data30);
        T32=str2num(data31);
        T3=[T31;T32];
        new_image3=mask_warping(new_image2,atlasnormimage,T3);
        %figure;imshow(new_image3);
        

        %% Forth_4
%         T4=[0.9896	-0.0005	16.0372
%             0.0018	0.9931	-3.2259];
        data40=dataread(filein,21);
        data41=dataread(filein,22);
        T41=str2num(data40);
        T42=str2num(data41);
        T4=[T41;T42];
        new_image4=mask_warping(new_image3,atlasnormimage,T4);
        %figure;imshow(new_image4);
        

        %% Fifth_5
%         T5=[0.9888	0.0016	15.1025
%             -0.0054	0.9982	8.0263];
        data50=dataread(filein,24);
        data51=dataread(filein,25);
        T51=str2num(data50);
        T52=str2num(data51);
        T5=[T51;T52];
        new_image5=mask_warping(new_image4,atlasnormimage,T5);
        out_file=fullfile(warped_mask,name2);
        %figure;imshow(new_image5);
        imwrite(new_image5, out_file);
        fprintf('a mask is %f secs \n',toc) 
        
        J=atlas1;
        thresh = graythresh(new_image5);     %�Զ�ȷ����ֵ����ֵ
        I2 = im2bw(new_image5,thresh); 
        I2=im2uint8(I2);
        I=cat(3,I2,I2,I2);
        [a2,b2,c2]=size(J);
        I=imresize(I,[a2 b2]); 
        % J=imresize(J,[1023 1500]); 
        J16=uint16(J); I16=uint16(I);
        L=immultiply(I16,J16); 
        L1=double(L);
        new_name2=[new_name2,'.png'];
        out_add=fullfile(add_mask,new_name2); 
        imwrite(L, out_add);
    end
    fprintf('Time for image warping is %f secs \n',toc) 
    catch
    end
end
%% read txt_file
function dataout=dataread(filein,line)
fidin=fopen(filein,'r');
%fidout=fopen(fileout,'w');
nline=0;
while ~feof(fidin) 
tline=fgetl(fidin); 
nline=nline+1;
if nline==line
%fprintf(fidout,'%s/n',tline);
dataout=tline;
end
end
fclose(fidin);
%fclose(fidout);
end