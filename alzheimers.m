load alzheimers\ad_data.mat
%parameters = [1e-8; 1e-5; 5e-5; 1e-4; 5e-4; 1e-3; 5e-3; 1e-2; 5e-2;0.1 ;0.2; 0.3]; % These parameters corresponds to figure NewSubsetNumFeatures 
parameters = [1e-8; 0.01; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];% These parameters corresponds to figure Num_Features 
accuracy = zeros(12,1);
features = zeros(12,1);

for i = 1:12
    par = parameters(i);
    [w, c] = logistic_l1_train(X_train, y_train, par);
    prediction =sigmoid(-X_test * w + c);
    [X,Y,T,AUC] = perfcurve(y_test, prediction, 1);
    accuracy(i) = AUC ;
    w(w~=0) = 1;
    features(i) = sum(w);
end

figure(1)
plot(parameters, accuracy, '-*');
xlabel('Parameters');
ylabel('Accuracy');

figure(2)
plot(parameters, features, '-*');
xlabel('Parameters');
ylabel('Number of Features');

function sig = sigmoid( input )
    sig = 1./ (1 + exp(input));
end