function classifier = getCNN(X, y, maxEpochs)
if nargin < 3
    maxEpochs = 500;
end
validateattributes(maxEpochs, {'numeric'}, {'scalar', 'integer', 'positive'});
height = size(X,2);
y = categorical(y);
noOfClasses = length(unique(y));
for i = 1:size(X,1)
    data(:,:,:,i) = X(i,:)';
end

y = y';

if height > 5
   layers = [
        imageInputLayer([height 1 1])
        convolution2dLayer([min(2,height) 1],max(1,height-1),'Stride',1)
        batchNormalizationLayer
        fullyConnectedLayer(100)
        reluLayer
        fullyConnectedLayer(100)
        reluLayer
        fullyConnectedLayer(noOfClasses)
        softmaxLayer
        classificationLayer
        ];
else
    layers = [
        imageInputLayer([height 1 1])
        convolution2dLayer([min(2,height) 1],max(1,height-1),'Stride',1)
        batchNormalizationLayer
        fullyConnectedLayer(100)
        reluLayer
        fullyConnectedLayer(noOfClasses)
        softmaxLayer
        classificationLayer
        ];
end

options = trainingOptions('sgdm',...
    'MaxEpochs',maxEpochs, ...
    'Verbose',false);


classifier = trainNetwork(data, y', layers, options);

end

