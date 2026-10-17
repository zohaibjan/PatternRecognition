function model = getCIFAR(X, y, maxEpochs)
%GETCIFAR Historical name: this network expects flattened 96x96 RGB images.
if nargin < 3, maxEpochs = 30; end
validateattributes(maxEpochs,{'numeric'},{'scalar','integer','positive'});
X = cifarImages(X);
y = categorical(y);
netDepth = 4; % netDepth controls the depth of the convolutional blocks
netWidth = 96; % netWidth controls the number of filters in a convolutional block

layers = [
    imageInputLayer([96 96 3])
    convolutionalBlock(netWidth,netDepth)
    maxPooling2dLayer(2,'Stride',2)
    convolutionalBlock(2*netWidth,netDepth)
    maxPooling2dLayer(2,'Stride',2)
    convolutionalBlock(4*netWidth,netDepth)
    averagePooling2dLayer(8)
    fullyConnectedLayer(numel(categories(y)))
    softmaxLayer
    classificationLayer
    ];

miniBatchSize = 2; % * numberOfWorkers;
initialLearnRate = 1e-1 * miniBatchSize/256;

options = trainingOptions('sgdm', ...
    'InitialLearnRate',initialLearnRate, ... % Set the initial learning rate.
    'MiniBatchSize',miniBatchSize, ... % Set the MiniBatchSize.
    'Verbose',true, ... % Do not send command line output.
    'L2Regularization',1e-10, ...
    'MaxEpochs',maxEpochs, ...
    'Shuffle','every-epoch', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',25);

model = trainNetwork(X, y,layers,options);

end

function layers = convolutionalBlock(numFilters,numConvLayers)
    layers = [
        convolution2dLayer(3,numFilters,'Padding','same')
        batchNormalizationLayer
        reluLayer
    ];

    layers = repmat(layers,numConvLayers,1);
end