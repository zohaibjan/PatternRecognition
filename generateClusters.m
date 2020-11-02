function genClusters = generateClusters(train , params)
totalClusters = 1;
genClusters = {};
dataClasses = unique(train(:,end))';
try
if totalClusters == 1
    [clusterIds, C, sum, D] = kmeans(zscore(train(:,1:end-1)), 1);
    genClusters{totalClusters}.train = train(find(clusterIds == 1), :);
    genClusters{totalClusters}.centroid = C(1,:);
    totalClusters = totalClusters + 1;
end

for i=1:length(dataClasses)
    Xtrain = train(train(:,end) == dataClasses(i),:);
    if size(Xtrain, 1) < params.eachClass
        continue;
    end
    [clusterIds, C, sum, D] = kmeans(zscore(Xtrain(:,1:end-1)), params.eachClass);
    for j=1:params.eachClass
        genClusters{totalClusters}.train = Xtrain(find(clusterIds == j), :);
        genClusters{totalClusters}.centroid = C(j,:);
        totalClusters = totalClusters + 1;
    end
end
catch exc 
    disp(fprintf('\nProblem with %s \n%s\n', params.p_name, exc.identifier));
end
end


 