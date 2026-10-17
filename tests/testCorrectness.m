function tests = testCorrectness
%TESTCORRECTNESS Synthetic boundary checks; no fitted learner or dataset needed.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
testCase.TestData.originalPath = path;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root, fullfile(root, 'tests', 'fixtures'), '-begin');
end

function teardownOnce(testCase)
path(testCase.TestData.originalPath);
end

function testNumericCategoryNames(testCase)
verifyEqual(testCase, numericLabels(categorical([0; 10; 20; 10])), [0; 10; 20; 10]);
verifyEqual(testCase, numericLabels({'-1'; '10'}), [-1; 10]);
end

function testRejectInvalidLabels(testCase)
verifyError(testCase, @() numericLabels({'person'}), 'Research:InvalidLabels');
verifyError(testCase, @() numericLabels([1; NaN]), 'Research:InvalidLabels');
end

function testVoteTiesAndNonconsecutiveLabels(testCase)
verifyEqual(testCase, votePredictions([10 20; -1 -1; 0 10]), [10; -1; 0]);
end

function testRejectEmptyAndFailedVotes(testCase)
verifyError(testCase, @() votePredictions(zeros(3, 0)), 'Research:InvalidPredictions');
verifyError(testCase, @() votePredictions([1 NaN]), 'Research:InvalidPredictions');
end

function testFeatureOnlyCentroidsAndRowCounts(testCase)
% More feature columns than samples must not inflate the augmentation count.
train = [0 0 0 0 0 0 0; 10 0 0 0 0 0 10; 11 0 0 0 0 0 10];
clusters{1}.train = train;
clusters{2}.train = train(1, :);
clusters{1}.centroid = [999 999]; % Legacy coordinates are deliberately ignored.
clusters{2}.centroid = [999 999];
[balanced, centers] = balanceClusters(clusters, train);
verifyEqual(testCase, size(centers{2}), [1 6]);
verifyEqual(testCase, centers{2}, zeros(1, 6));
verifyEqual(testCase, balanced{2}, train(1:2, :));
end

function testTinyClusterSearchProducesFiniteFeatureCentroids(testCase)
params = struct();
[clusters, count] = generateClustersV2([0 10; 1 10], params);
verifyEqual(testCase, count, 1);
verifyEqual(testCase, numel(clusters), 2);
verifyEqual(testCase, clusters{1}.centroid, .5);
verifyEqual(testCase, clusters{2}.centroid, .5);
end
