function w = bmrmUpdateModelL2(A, B, lambda, v, inside_cccp)
% w = bmrmUpdateModelL2(A, B, lambda, v, inside_cccp) updates the model
% parameter w using linear lower bound A and B. lambda is the parameter for
% the L2 regularization. v is a non-zero vector and inside_cccp is 1 when
% calling bmrm inside a cccp procedure; they are zero otherwise.
% 
% Input:
%       A, B:       gradients and offsets as the linear lower bound
%       lambda:     parameter for the L2 regularization
%       v:          linear upperbound (for learning from partial annotations)
%       inside_cccp:    indicating whether bmrm is inside a cccp procedure
% 
% Output:
%       w:          updated parameter
% 
% This code is part of PLEASD toolbox. 
% Copyright (C) 2012 Xinghua Lou (xinghua.lou@gmail.com)
%

A = cell2mat(A);
B = cell2mat(B);

% formulate the QP problem
H = (A'*A) ./ lambda;
f = (((v'*A) ./ lambda) - B)';

% find QP solver
if exist('cplexqp', 'file') ~= 0
    qp_solver = @cplexqp;
elseif exist('quadprog', 'file') ~= 0
    qp_solver = @quadprog;
else
    error('Cannot find any QP solver!');
end

% solve the QP acccordingly: note that pure bmrm run and bmrm inside a 
% cccp run have different constraint formulations.
if ~inside_cccp
    Aineq = ones(1, length(f)); bineq = 1;
    lb = zeros(length(f), 1);
    [alpha, fval, exitflag, output] = qp_solver(...
        H, f, Aineq, bineq, [], [], lb, []);
else
    Aeq = ones(1, length(f)); beq = 1;
    lb = zeros(length(f), 1);
    [alpha, fval, exitflag, output] = qp_solver(...
        H, f, [], [], Aeq, beq, lb, []);
end

if exitflag <=0
    disp(output);
    disp(fval);
    error('Error occured in QP solver. Problem can not be solved.');
end

% transform to prime solution and update the intermediate result matrix
w = - (v + A*alpha) ./ lambda;
