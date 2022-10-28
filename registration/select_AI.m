function [AIimage] = select_AI(img_name)
base_dir=pwd; 
kmeans_6888=fullfile(base_dir,'data','kmeas_conv_6888.xlsx');
all_image=xlsread(kmeans_6888,'Sheet1','A1:A6663');
[img_idex,~]=find(all_image==double(img_name));
all_classier=xlsread(kmeans_6888,'Sheet1','F1:F6663');
classier=all_classier(img_idex);
all_cor=xlsread(kmeans_6888,'Sheet2','F1:F118');
cor_index=num2str(all_cor(classier+1));
cor_suffix=[cor_index,'.jpg'];

AI_dir= fullfile(base_dir,'data','AI_684');
disp(cor_suffix);
AIimage=imread(fullfile(AI_dir,cor_suffix));

end

