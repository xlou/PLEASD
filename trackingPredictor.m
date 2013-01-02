function y_out = trackingPredictor(d, w, augment, search_space)
% y_out = trackingPredictor(d, w, augment, search_space) predicts the most
% likely output y_out for sample d and parameter w. 
% 
% Input:
%       d:          data sample
%       w:          model parameter
%       augment:    =  0, no loss augment, i.e. <Psi, w> only
%                   =  1, loss augment, i.e. <Psi, w> + Delta(y_dot, y_hat)
%                   = -1, loss augment, i.e. <Psi, w> - Delta(y_dot, y_hat)
%       search_space:   =  0, entire output space
%                       =  1, space consistent with y_hat (for partial annotatoin)
%                       = -1, space inconsistent with y_hat (for partial annotatoin)
% 
% Output:
%       y_out:      predicted output
% 
% Hints:
% y_out = trackingPredictor(d, w) assumes augment = 0 and search_space = 0.
%
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

if nargin < 3
    augment = 0;
    search_space = 0;
end

% create the main objective function
w = mat2cell(w, colsInCell(d.x), 1);
f = cell(size(d.x));
for i = 1:length(d.x)
    f(i) = {d.x{i} * w{i}};
end
f = cell2mat(f);

% add the augment problem, if necessary
if augment ~= 0
    loss_vec = -(cell2mat(d.y_dot) == 1);
    loss_vec = loss_vec ./ sum(cell2mat(d.y_dot) == 1);
    f = f + augment * loss_vec;
end

% formulate additional constraints on the search space, if necessary
Aineq = []; bineq = [];
Aeq = d.omega.Aeq; beq = d.omega.beq;
if search_space == 1
    y_dot_vec = cell2mat(d.y_dot);  % vectorize
    Aeq_ = 2*(y_dot_vec - 0.5)'; Aeq_(y_dot_vec == -1) = 0;
    beq_ = sum(y_dot_vec(y_dot_vec ~= -1));
    Aeq = [Aeq; Aeq_]; beq = [beq; beq_];
elseif search_space == -1
    y_dot_vec = cell2mat(d.y_dot);  % vectorize
    Aineq_ = 2*(y_dot_vec - 0.5); Aineq_(y_dot_vec == -1) = 0; Aineq_ = Aineq_';
    bineq_ = sum(y_dot_vec(y_dot_vec ~= -1)) - 1;
    Aineq = [Aineq; Aineq_]; bineq = [bineq; bineq_];
end

% find bilp solver
if exist('cplexbilp', 'file') ~= 0
    bilp_solver = @cplexbilp;
elseif exist('bintprog', 'file') ~= 0
    bilp_solver = @bintprog;
else
    error('Cannot find any binary ILP solver!');
end

% call cplex ILP solver to solve the inference problem
[y_out, fval, exitflag, output] = bilp_solver(-f, Aineq, bineq, Aeq, beq);
if exitflag <=0
    disp(output);
    disp(fval);
    error('Error occured in ILP solver. Problem can not be solved.');
end

% convert vector x to cell
y_out = mat2cell(y_out, rowsInCell(d.x), 1);

end
