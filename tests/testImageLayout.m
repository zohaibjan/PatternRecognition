function tests = testImageLayout
tests = functiontests(localfunctions);
end
function testSamplesNeverInterleave(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
rows = [zeros(1,96*96*3);255*ones(1,96*96*3);128*ones(1,96*96*3)];
images = cifarImages(rows);
verifySize(testCase,images,[96,96,3,3]);
verifyEqual(testCase,images(:,:,:,1),zeros(96,96,3));
verifyEqual(testCase,images(:,:,:,2),ones(96,96,3));
verifyEqual(testCase,images(:,:,:,3),128/255*ones(96,96,3),'AbsTol',1e-12);
end
function testWrongImageWidthRejected(testCase)
verifyError(testCase,@() cifarImages(zeros(2,32*32*3)),'Research:ImageDimensions');
end
