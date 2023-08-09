clc, clear, close all
%% LDA succeed, PCA fail Data

%class 1 data 
cls1_data = [2.93 6.634; 2.53 7.79; 3.57 5.65; 3.16 5.47]; %succeed data
%cls1_data = [2.93 1.5; 4.5 5; 3.5 0.5; 2.4 5.1]; %fail data
cls1_n = length(cls1_data);
%class 2 data
cls2_data = [2.58 4.44; 2.16 6.22; 3.27 3.52; 2.2 3]; %succeed data
%cls2_data = [2.58 4.44; 3.9 6.22; 4.4 1; 2.2 3]; %fail data
cls2_n = length(cls2_data);

%mean
cls1_mean = mean(cls1_data); 
cls2_mean = mean(cls2_data);
all_mean =  mean([cls1_data; cls2_data]);

plot(cls1_data(:, 1), cls1_data(:, 2), '*r');
hold on 
plot(cls2_data(:, 1), cls2_data(:, 2), '*b');
title("2-class LDA Analysis", 'FontSize', 24);
xlabel("Feature 1", 'FontSize', 24);
ylabel("Feature 2", 'FontSize', 24);

%% Class Mean and All Mean differences
x1 = cls1_mean - all_mean;
x2 = cls2_mean - all_mean;

Sb = (cls1_n / (cls1_n + cls2_n)) * x1' * x1 + ...
    (cls2_n / (cls1_n + cls2_n)) * x2' * x2;

y1 = 0;
for i = 1:cls1_n
    y1 = y1 + (cls1_data(i, :) - cls1_mean)' * (cls1_data(i, :) - cls1_mean); 
end

y2 = 0;
for i = 1:cls2_n
   y2 = y2 + (cls2_data(i, :) - cls2_mean)' *(cls2_data(i, :) - cls2_mean); 
end

Sw = (cls1_n / (cls1_n + cls2_n)) * y1 + ...
    (cls2_n / (cls1_n + cls2_n)) * y2;

%% Find reference vector to project onto the data points 
[eig_vec, eig_val] = eig( inv(Sw) * Sb );
[largest_eig_val, index] = max( max(eig_val) );
vector = eig_vec(:, index); 

new1_data = cls1_data*vector;
new2_data = cls2_data*vector;

%% Plot vector & Projected points

%add vector to the plot
m = vector(2) / vector(1); %slope of vector
plot([0, 10], [0, 10 * m], '-g');

%Project points for class 1
for i = 1:cls1_n
    newx = (cls1_data(i, 1) + m * cls1_data(i, 2)) / (m^2 + 1);
    newy = m * newx;
    plot(newx, newy, '*r'); 
    plot([cls1_data(i, 1), newx], [cls1_data(i, 2), newy], '-r');
end

%Project points for class 2
for i = 1:cls2_n
    newx = (cls2_data(i, 1) + m*cls2_data(i, 2))/(m^2 + 1);
    newy = m*newx;
    plot(newx, newy, '*b'); 
    plot([cls2_data(i, 1), newx], [cls2_data(i, 2), newy], '-b');
end

%% Classify new point using LDA
test_data = [4.81, 3.46];
plot(test_data(1), test_data(2), '*g');
result = test_data*vector; 

%Project test_data to the reference vector
projected_x = (test_data(1) + m*test_data(2)) / (m^2 + 1);
projected_y = m*projected_x;
plot(projected_x, projected_y, '*k')

plot([test_data(1), projected_x], ...
    [test_data(2), projected_y], '--k');

%Difference between projected test point VS projected class1 data points
temp1 = new1_data - result;

%Difference between projected test point VS projected class2 data points
temp2 = new2_data - result;

if ( min(abs(temp1) ) < min( abs(temp2)) )
    prediction = 'class1(red)';
else
    prediction = 'class2(blue)';
end

legend('class 1', 'class 2');