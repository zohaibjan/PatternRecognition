function program = mainProgram()

% Problem = dataSetNames();                 % Get list of dataset names
addpath(genpath('P-Data'));
% Problem = {'thyroid', 'splice','steelplates','whitewine',...
%           'yeast', 'wall-following','optical','page-blocks',...
%           'adult','pendigits','letter-recognition','balance', ...
%           'banknote', 'breast-cancer-wisconsin','haberman',...
%           'iris', 'pima_diabetec', 'statimag',...
%           'teaching', 'vehicle', 'wdbc', 'wine',...
%           'hepatitis', 'fertility', 'zoo', 'appendicitis',...
%           'hayes-roth', 'planrelax', 'segment2', 'statimag' ,...
%           'glass', 'thyroid', 'spectfheart', 'heart', 'bupa'};

 
% Problem = {'thyroid', 'splice','steelplates','whitewine',...
%           'yeast', 'wall-following','optical','page-blocks',...
%             'ecoli', 'hayes-roth','vehicle'};

Problem = {'shuttle'};

%% Model SETTINGS
params.numOfFolds = 10;                   % Create CROSS VALIDATION FOLDS
%% MAIN LOOP
for i=1:length(Problem)
        p_name = Problem{i}
        results = runTraining(p_name, params);
        saveResults(results, p_name);
end

end




