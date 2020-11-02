function results = singleTrain(p_name , params)
warning('off','all');

mVote = [];

%% ITERATE OVER THE NUMBER OF FOLDS
for f=1:10
    data=load([pwd,filesep,'DTE',filesep,p_name,filesep,p_name,'-CV-tr-', num2str(f)]);
    data = [data.dtrX, data.dtrY];
    data = rmmissing(data);
    X= dataNormalize(data(:,1:end-1),2); 
    Y=data(:,end);
    data = [X Y];
    classifiers= {};
    classifierIndex = 1;
    
    %% SEPARATE TRAIN / TEST DATA PER FOLD
    cv = cvpartition(data(:,end), 'holdout', 0.1);
    idxs = cv.test;
    testData = data(idxs,:);
    trainData = data(~idxs, :);
    
    trainX = trainData(:, 1:end-1);
    trainy = trainData(:, end);
    
    testX = testData(:, 1:end-1);
    testy = testData(:, end);  
     
    classifiers{classifierIndex} = getCNN(trainX, trainy);    
    mVote(f) = majVote(classifiers, testX, testy)      
    
end
results.majVote = mean(mVote);
results.stdDev = std(mVote);

end




