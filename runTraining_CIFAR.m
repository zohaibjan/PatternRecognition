function results = runTraining_CIFAR(p_name, params)
%RUNTRAINING_CIFAR Legacy 96x96 RGB experiment; train and test remain separate.
loaded = load(fullfile(pwd,'P-Data',p_name));
if ~isfield(loaded,'train') || ~isfield(loaded,'test')
    error('Research:ImageDataset','Dataset must contain train and test labelled matrices.');
end
trainingData = loaded.train;
testData = loaded.test;
clusters = generateClusters(trainingData,params);
[balanced,centroids] = balanceClusters(clusters,trainingData);
models = cell(1,numel(balanced));
for k = 1:numel(balanced)
    rows = balanced{k};
    models{k} = getCIFAR(rows(:,1:end-1),rows(:,end));
end
X = testData(:,1:end-1);
centers = vertcat(centroids{:});
labels = zeros(size(X,1),1);
routes = zeros(size(X,1),1);
for k = 1:size(X,1)
    % First centroid breaks equal-distance ties deterministically.
    [~,routes(k)] = min(sum(bsxfun(@minus,centers,X(k,:)).^2,2));
end
for k = 1:numel(models)
    rows = routes==k;
    if any(rows)
        labels(rows) = numericLabels(classify(models{k},cifarImages(X(rows,:))));
    end
end
results = mean(labels == testData(:,end));
end
