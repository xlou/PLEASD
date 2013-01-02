function println(verbose, varargin)
% println(verbose, varargin) prints text to console with an additional '\n'
% at the end.
% 
% Input:
%       verbose:    = 1, println text to console
%                   = 0, skip this printing
%       varargin:   same as arguments in sprintf
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

if ischar(verbose)       % no indication of verbose
    varargin = {verbose, varargin{:}};
elseif verbose ~= 1
    return ;
end
varargin(1) = {sprintf('%s\n', varargin{1})};
fprintf(1, varargin{:});