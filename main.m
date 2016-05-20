
% size of training, validation and testset
set_sizes = [60 20 20];

%directory holding the input of different people
input_dir = 'data';

%input type of images
type = '.BMP';

%number of hidden nodes used
numhidden = 20;

rng(1234);
[wh, wo, error] = training(set_sizes, input_dir, type, numhidden);



fig1 = figure(1);
%clear figure first
clf;
%plot cycle error, add labels and change fontsize
set(gca,'fontsize',16)
hold on

plot(error)
xlabel('cycle iteration')
ylabel('cycle error')
title('cycle error curve for multilayer backpropagation')

