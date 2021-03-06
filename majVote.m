function acc=majVote(classifiers, X, y )
c = length(classifiers);
decisionMatrix = zeros(length(X(:,1)), c);
index = 1;
for i=1:c
    decisionMatrix(:,index) = getCNNPred(classifiers{1,i}, X);
    index = index + 1;
end
decisionMatrix = categorical(mode(decisionMatrix, 2));
y = categorical(y);
acc = mean(decisionMatrix == y);
end

