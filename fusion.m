function acc=fusion(ensemble, centroids, test)
decisionMatrix = zeros(size(test,1),1);

for i=1:size(test,1)
    distances = zeros(1, length(centroids));
    for j =1:length(centroids)
        distances(j) = norm(double(test(i,1:end-1) - centroids{1,j}));
    end
    index = find(distances == min(distances));
    if length(index) > 1
        preds = zeros(1, length(index));
        for k=1:length(index)
            preds(1,k) = getCNNPred(ensemble{1,index(k)}, test(i,1:end-1));
        end
        decisionMatrix(i,:) = mode(preds,2);
    elseif length(index) == 1
        decisionMatrix(i,:) = getCNNPred(ensemble{1,index}, test(i,1:end-1));
    end
end
acc = mean(decisionMatrix == test(:,end));

end