function results = runTraining(p_name , params)
warning('off','all');
params.p_name = p_name;
mVote = [];
e_mVote = [];
clusterBased = [];
bestCNN = [];

%% ITERATE OVER THE NUMBER OF FOLDS
for f=1:10
    data=load([pwd,filesep,'DTE',filesep,p_name,filesep,p_name,'-CV-tr-', num2str(f)]);
    data = [data.dtrX, data.dtrY];

    testdata=load([pwd,filesep,'DTE',filesep,p_name,filesep,p_name,'-CV-ts-', num2str(f)]);
    testX = testdata.dtsX;
    testy = testdata.dtsY;
    
    X = data(:, 1:end-1);
    Y = data(:,end);
    data = [X Y];
    classifiers= {};
    classifierIndex = 1;
         
    cv = cvpartition(data(: , end), 'holdout', 0.1);
    idxs = cv.test;
    validationData = data(idxs,:);
    trainData = data(~idxs, :);
                  
    trainX = trainData(:, 1:end-1);
    trainy = trainData(:, end);
    
    valX = validationData(:, 1:end-1);
    valy = validationData(:, end);
    
    allClusters = generateClustersV2([trainX, trainy], params);
    [allClusters, centroids] = balanceClusters(allClusters, [trainX trainy]);
    
    
    for c=allClusters
        X = c{1,1}(:, 1:end-1);
        y = c{1,1}(:, end);
        classifiers{classifierIndex} = getCNN(X, y);
        classifierIndex = classifierIndex + 1;
    end
    
    ensemble = ensembleSelection(classifiers, valX, valy);
    
    mVote(f) = majVote(classifiers, testX, testy);
    e_mVote(f) = majVote(ensemble, testX, testy);
    clusterBased(f) = fusion(classifiers, centroids, [testX testy]);
   
    
end
results.majVote = mean(mVote);
results.mstd_dev = std(mVote);
results.clusterBased = mean(clusterBased);
results.cstd_dev = std(clusterBased);
results.e_mVote = mean(e_mVote);
results.e_mVote_dev = std(e_mVote);

disp(fprintf('%s:  Majvote :  %4f \nFusion :  %4f\n\n\n',p_name, results.majVote, results.clusterBased));
end




