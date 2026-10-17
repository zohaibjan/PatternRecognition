function images = cifarImages(X)
%CIFARIMAGES Convert rows of flattened 96x96 RGB pixels without mixing samples.
validateattributes(X,{'numeric'},{'2d','nonempty','real','finite','>=',0,'<=',255});
if size(X,2) ~= 96*96*3
    error('Research:ImageDimensions','Expected one 27648-value, column-major RGB image per row.');
end
images = reshape(double(X.')/255,[96,96,3,size(X,1)]);
end
