function [ hweigth, oweigth, errors ] = training( set_sizes, in_dir, type, numhidden)
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
    %expected output
    D = [];

    for person_idx = 1:length(persons)
        %only process valid directories
        curperson = char(persons(person_idx));
        curtraindata = dir(strcat(curperson,  '/*', type));
        curdata=100;
        
        for i = 1:curdata%length(curtraindata)
            filename = strcat( strcat(curperson,  '/', curtraindata(i).name));
            %load image
            img = imread(filename);
            %normalize image to value between 0 and 1
            img_norm = mat2gray(img);
            %round so we get a binary image
            img_bin = round(img_norm);
            %transform matrix to single row vector
            X(:, (person_idx-1)*curdata + i) = img_bin(:)';


            %form expected output matrix all entries except person are -1
            D(:, (person_idx-1)*curdata + i) = -ones(length(persons),1);
            D(person_idx, (person_idx-1)*curdata + i) =  1;
        end
    end

    trainset = set_sizes(1) * curdata / 100

    %append augmented -1 for bias 
    X = [X; -ones(size(X(1,:)))];

    %initial weight vectors
    w_output = rand(length(persons),numhidden+1);

    w_hidden = rand(numhidden,length(X(:,1)));



    %learning constant
    n = 0.8;


    % doing 500 cycles of training the patterns
    for c=1:50

        delta_bar = [];


        %one cycle of update process all input data
        for i=1:length(X(1,:))
            %first do a forward propagation step to calculate y so we can
            %finally calculate the predicted value of z to calculate the error
            % (transpose (w_hidden * X(:,i)) otherwise the exponential function
            % does not give the correct result)
            y = ( 2./( 1+ exp(-((w_hidden * X(:,i))')))) - 1;
            %assign -1 to augmented input for y
            y = [y -1];

            %calc with bipolar activation function for output layer
            z = ( 2./( 1+ exp(-((w_output * y')')))) - 1;

            e = D(:,i)' - z;
            %calculate error pattern for later calculation of cycle error
            EP(c,i) = 0.5*sum(e)^2;
            df = 0.5*(1-z.^2);
            %calculate the delta for updating w_output
            delta = e .* df;

            %calculate hidden layer delta for updating w_hidden
            df = 0.5*(1-y.^2);        
            delta_bar = df .* (delta * w_output);

            %update the output w
            w_output = w_output + n*delta'*y;

            %update the hidden w
            w_hidden = w_hidden + n.*delta_bar(1:end-1)'*X(:,i)';
        end
        %sum up the pattern errors to calculate the final cycle error
        EC(c) = sum(EP(c,:));


    end
    hweigth = w_hidden;
    oweigth = w_output;
    errors = EC; 

end




