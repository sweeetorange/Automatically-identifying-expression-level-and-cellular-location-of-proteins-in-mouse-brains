function  [output_image,finalCorresNum,thresh4] = final_registration(new_image,edge_atlas,atlasorientation,flag,alpha, edgepath)

disp('Final Registration')

I=new_image(:,:,flag); %imtool(I) 
%I = new_image;
% I0=new_image(:,:,flag);  %imtool(I) 
% %I1=new_image(:,:,1);
% I2=new_image(:,:,3);
% a=ave_gray(I0);
% b=ave_gray(I2);
% if a<b
%     I=I2;
% else
%     I=I0;  
%     fprintf('channel=green')%select the green or blue channel
% end
    
% median filtering
G0=medfilt2(I,[20 20]); %imtool(G0)  

% filtering & smoothening   
hgaus=fspecial('gaussian',12,2); %figure,surf(hgaus)    
                                                         
G1=imfilter(G0,hgaus); %imtool(G1)

% Computed the threshold for hysterisis to be used for canny edge  
thresh4 = threshold_Histogram(G1,2,false);                    % Very important   

% edge detection
Inew=edge(G1,'canny',[0 thresh4]); %imtool(Inew)            %this threshold can be varied depending on the edges you want
Inew=remove_HoriVeriLines(Inew,50); %imtool(Inew)

% connected componenet of >300 pixels.
[Inew1,~] = largestConnectedComponent(Inew,200,true);

imwrite(Inew1,fullfile(edgepath,'5_FinalRegis_edgeImage_whoseNormalVectorsAreMatched.jpg')); % storing the edge image for testing

% normal vector computation for image & atlas
[ximage_normal,yimage_normal]=normal_vector(Inew1,false);

% computing orientation of normal vectors.(angles of the normal vectors)
imageorientation=atand(yimage_normal./ximage_normal);

% final correspondence 
[A,bx,by]=Final_PointCorrespondence(imageorientation,atlasorientation);     % distance for final correspondece is set to 50 and angle 1

% identifying points from damaged areas.
pointsTobeRemoved = damagedPoints(G1,thresh4,alpha);

h1 = figure();set(gcf,'Visible', 'off');                                    % storing the damaged poitns for testing
imshow(Inew1);hold on
plot(pointsTobeRemoved(:,1),pointsTobeRemoved(:,2),'c*','MarkerSize',4);
saveas(h1,fullfile(edgepath,'6_Damaged_PointsIdentified.tif'));
close(h1)

d = ismember([A(:,1) A(:,2)],pointsTobeRemoved,'rows');  % removing points from symmetry
A(d,:)=[];
bx(d,:)=[];
by(d,:)=[];

finalCorresNum=size(A,1);

% final warping
output_image = Laplace_warping(A,bx,by,edge_atlas,new_image);

end

