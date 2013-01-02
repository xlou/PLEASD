function rows = rowsInCell(C)
% rows = rowsInCell(C) returns the number of rows of matrics in a cell.
%
% Input:
%       C:          a cell object
% 
% Output:
%       rows:       number of rows of matrics in C
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

rows = zeros(size(C));
for i = 1:size(C)
    rows(i) = size(C{i}, 1);
end
