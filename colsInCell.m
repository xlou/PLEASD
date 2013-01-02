function cols = colsInCell(C)
% cols = colsInCell(C) returns the number of columns of matrics in a cell.
%
% Input:
%       C:          a cell object
% 
% Output:
%       cols:       number of columns of matrics in C
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

cols = zeros(size(C));
for i = 1:size(C)
    cols(i) = size(C{i}, 2);
end

end