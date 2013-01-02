function v = getField(S, name, vDef)
% v = getField(S, name, vDef) returns the value of the field in struct,
% e.g. S.(name); if S doesnot contain a field with this name, it returns
% the default value vDef instead.
%
% Input:
%       S:          a struct object
%       name:       name of the field
%       vDef:       default value
% 
% Output:
%       v:          output value of the field
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

if isfield(S, name)
    v = S.(name);
else
    v = vDef;
end
