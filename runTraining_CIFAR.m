function results = runTraining_CIFAR(p_name , params)
warning('off','all');

accuracy = [];
optaccuracy = [];
ensemble = {};
classifierIndex = 1;
load([pwd,filesep,'P-Data',filesep,p_name]);


noOfClasses = unique(train(:,end))';
clusters = {};
totalClusters = 1;

clusters = generateClusters(train, params);
[balancedClusters, centroids] = balanceClusters(clusters, train);

for c=balancedClusters
    X = c{1,1}(:, 1:end-1);
    y = (c{1,1}(:, end));
    ensemble{1, classifierIndex} = getCIFAR(X, y);
    classifierIndex = classifierIndex + 1;
end

decisionMatrix = zeros(length(test),1);

for i=1:length(test)
    distances = zeros(1, length(centroids));
    for j =1:length(centroids)
        distances(j) = norm(test(i,:) - centroids{1,j});
    end
    index = find(distances == min(distances));
    decisionMatrix(i,:) = predict(ensemble{1,index}, test(i,1:end-1));
end

groundTruth = test(:, end);
results = mean(decisionMatrix == groundTruth);

end