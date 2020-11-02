function rs = saveReults(results, p_name)
if (exist([pwd filesep 'results.csv'], 'file') == 0)
    fid = fopen([pwd filesep 'results.csv'], 'w');
    fprintf(fid, '%s,%s,%s, %s, %s, %s, %s \n', ...
        'Data Set','MajVote Accuracy', 'Std dev.', 'Cluster based accuracy', 'Std. dev', ...
        'ES Accuracy', 'ES Std Dev.');
elseif (exist([pwd filesep 'results.csv'], 'file') == 2)
    fid = fopen([pwd filesep 'results.csv'], 'a');
end
fprintf(fid, '%s, ', p_name);
fprintf(fid, '%f,%f, %f, %f, %f, %f\n', ...
    results.majVote, results.mstd_dev, results.clusterBased,...
    results.cstd_dev, results.e_mVote, results.e_mVote_dev);
fclose(fid);
end



