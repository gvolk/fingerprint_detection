function [ weight_vector, errors ] = training( set_sizes, in_dir, type)
%function for training the neural network

data_dirs = dir(in_dir);
persons = {}
i = 1;

%first extract the names of the directories
for idx = 1:length(data_dirs)
    %only process valid directories
    if( length(data_dirs(idx).name) > 2 )
       persons(i) = cellstr(strcat(in_dir , '/' , data_dirs(idx).name));
       i = i+1;
    end
end

%input vector for training data each row representing a person 
X = [];

for person_idx = 1:length(persons)
    %only process valid directories
    curperson = char(persons(person_idx));
    curtraindata = dir(strcat(curperson,  '/*', type));
    
    for i = 1:length(curtraindata)
        filename = strcat( strcat(curperson,  '/', curtraindata(i).name));
        %load image
        img = imread(filename);
        %normalize image to value between 0 and 1
        img_norm = mat2gray(img);
        %round so we get a binary image
        img_bin = round(img_norm);
        %transform matrix to single row vector
        X(person_idx, i, :) = img_bin(:);
    end
end





