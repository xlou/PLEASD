function Psi = trackingJointFeature(x, y)
% Psi = trackingJointFeature(x, y) return a vector of features that
% describes the compatibility of input x and output y
% 
% Input:
%       x:          structured intput
%       y:          structured output
%  
% Output:
%       Psi:        must be a double vector of Dx1 where D is the size of
%                   the joint feature vector
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

% form the joint feature vector
Psi = cell(size(x));
for i = 1:length(x)
    Psi(i) = {(x{i})' * y{i}};
end
Psi = cell2mat(Psi);
