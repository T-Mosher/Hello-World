% train on the ex1_multi data set using fminunc()
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);
[X mu sigma] = featureNormalize(X);
% Add intercept term to X
X = [ones(m, 1) X];
initTheta = zeros(size(X, 2), 1);
options = optimset('GradObj', 'on', 'MaxIter', 100);
[optTheta, functionVal, exitFlag, outp] = fminunc(@(t)(computeCostConjugateGrad(t, X, y)), initTheta, options);
%[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
disp(optTheta);
disp(functionVal);
disp(exitFlag);
disp(outp);

house_features = [1 ([1650 3] - mu) ./ sigma]';
price = theta_g' * house_features; % You should change this

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using fminunc):\n $%f\n'], price);
