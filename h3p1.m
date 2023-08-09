clc; close all; clear all;
%% 
%uniform distribution
x_norm = randi([0 100],1,100);
y_norm = randi([0 100],1,100);

%Gaussian distribution
x_gaus = normrnd(0,100/3, [1,100]);
y_gaus = normrnd(0,100/3, [1,100]);

%exponential distribution
mu1 = ones(1,100);
y_exp = exprnd(mu1);
x_exp = exprnd(mu1);

%Initial weight - 100 neurons (10x10) assigned a weight vector
w = normrnd(0, 1/3,  [10,10,2]);

%Initial learning rate (dynamic)
lr_initial = 1;

%%  create training data for different distributions
train_norm = [x_norm ; y_norm];
train_norm = train_norm';
train_gaus = [x_gaus ; y_gaus];
train_gaus = train_gaus';
train_exp = [x_exp ; y_exp];
train_exp = train_exp';

iteration = 1000;

%size of each neuron
width_initial = 15; %to show density estimation
t_width = iteration/log(width_initial);
%% test plot
%scatter(x_norm,  y_norm, '.');
train_data = train_exp; %change this among 3 training models (train_norm, train_gaus, train_exp)
y =  2.* ones(100,1);
%plotData(train_data, y);
scatter(x_exp,y_exp, '.');
title("2D-lattice SMO Plot for 100 neurons (iteration = 1000, exp dist.)", 'FontSize', 24);
xlabel("feature 1", 'FontSize', 24);
ylabel("feature 2", 'FontSize', 24);
hold on;

%% Repeat until convergence
for t = 1 : iteration

    %Decrease neighborhood size 2c
    width = width_initial*exp(-t/t_width);
    width_variance = width^2;

    %Decrease learning rate 2b
    lr = lr_initial*exp(-t/iteration);
    % Prevent learning rate becoming too small
    if lr < 0.05
        lr = 0.1;
    end
    %% find winner in eucd. distance
    distance = zeros(length(w),length(w));
    index = randi([1 length(x_norm)]); 

    for r = 1 : length(w)
        for c = 1:length(w)
            v = train_data(index, :) - reshape(w(r,c,:),1,size(w, 3));
            distance(r,c) = sqrt(v*v');
        end
    end

    [min_val, ind] = min(distance(:));
    [win_r, win_c] = ind2sub(size(distance), ind);
    %Winner decided

    %% Compute neighbor 2a1
    neighbor = zeros(length(w), length(w));

    for r = 1:length(w)
        for c = 1:length(w)
            if (r == win_r) && (c == win_c) %if this neuron is the one that won
                neighbor(r,c) = 1;
            else %if this neuron is  NOT the one that won
                dist = (win_r - r)^2 + (win_c - c)^2;
                neighbor(r,c) = exp(-dist / (2*width_variance));
            end
        end
    end

    %% Update Weights 2a2
    temp = zeros(length(w), length(w), size(w, 3));
    
     for r = 1:length(w)
        for c = 1:length(w)
            %update step by step, looping through rows and columns
            current_vector = reshape(w(r,c,:),1,size(w,3));

            temp(r,c,:) = current_vector + ...
                lr * neighbor(r,c) * (train_data(index, :) - current_vector);
        end
     end

     w = temp;

    %% Illustration
    fprintf('\n SMO Plotting ...\n')

    dot = zeros(length(w)*length(w), size(w,3));
    matrix = zeros(length(w)*length(w), 1);
    matrix_old = zeros(length(w)*length(w), 1);

    ind = 1;
    hold on;
    f1 = (figure(1));
    set(f1,'name',strcat('Iteration #',num2str(t)),'numbertitle','off');
   
    for r = 1:length(w)
        for c = 1:length(w)      
            dot(ind,:)=reshape( w(r,c,:), 1, size(w,3) );
            ind = ind + 1; 
        end
    end

    %plot
    for r = 1 : length(w)
        r1 = 1 + length(w) * (r-1);
        r2 = r * length(w);
        c1 = length(w)^2;

        temp1(2*r-1, 1) = plot(dot(r1:r2,1),dot(r1:r2,2), ...
            '--ro','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor', ...
            'g','MarkerSize',4);
        temp1(2*r,1) = plot(dot(r:length(w):c1,1),dot(r:length(w):c1,2), ...
            '--ro','LineWidth',2,'MarkerEdgeColor','g','MarkerFaceColor', ...
            'g','MarkerSize',4);

        temp2(2*r-1,1) = temp1(2*r-1,1);
        temp2(2*r,1) = temp1(2*r,1);

    end
    
    if t ~= iteration
        for r = 1:length(w)
            delete( temp2(2*r-1, 1));
            delete( temp2(2*r,1));
            drawnow;
        end
    end

    


    





end







