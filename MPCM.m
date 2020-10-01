%%% MPCM: an uncertainty evaluation method 
%   [g, meanOutput] = MPCM(symbols, distributions, ranges, degrees, G)
%   estimates the mean output of a system with dynamics described by G. 
%
%%% The inputs of the MPCM function include:
%       symbols: a set of uncertain input parameters defined using 'syms'
%                and saved in a 1xn vector, where n is the total of number 
%                of input parameters. The i-th element defines the i-th input 
%                parameter.
%       distributions: distributions (pdf) of the input parameters saved in a
%                      cell array. The i-th element defines the pdf of the
%                      i-th input parameter.
%       ranges: ranges of the input parameters saved in a nX2 matrix. The
%              i-th row defines the range of the i-th input parameters. 
%              The first column defines the minimum values of the input
%              parameters and the second column defines the maximum values
%              of the input parameters.
%       degrees: the maximum degrees of the uncertain input parameters in 
%                the reduced-order mapping g, saved in a 1xn vector. In 
%                particular, the maximum degree of the i-th input parameter 
%                in the reduced-order mapping g is ni-1, where ni is the
%                i-th element of the degrees vector. The M-PCM gaurantees
%                accurate mean output prediction for systems of each 
%                parameter's degree up to 2ni-1.                
%       G: system mapping/model. 
%
%%% The outputs of the MPCMOFFD function include:
%       g: reduced-order system mapping. 
%       meanoutput: mean output of the system G. 
%
%%% Example: find the mean output of a system with dynamics modulated by five
%          uncertain input parameters, {x1,x2,x3,x4,x5}, where x1 follows a
%          normal distribution, x2 follows an exponential distribution, and
%          x3, x4, x5 follow uniform distributions. The system mapping is
%          captured by G.
%      
%       syms x1 x2 x3 x4 x5;
%       symbols = [x1, x2, x3, x4, x5];
%       distributions = {1/sqrt(2*pi)/2*exp(-(x1-1)^2/2/2^2),exp(-x2),1, 10/9, 1/3};
%       ranges = [-inf, inf;0,inf;0,1; 0.1,1; -2,1]; 
%       degrees = [2, 2, 2, 2, 2];
%       G = 1.5 - 0.5*x1^2  - 2*x3^3 + 5*x1^3*x2 + x4+x5^2;
%       [g, meanOutput] = MPCMOFFD(symbols, distributions, ranges, degrees, G)
%
%%% References:
%       Y. Zhou, Y. Wan, S. Roy, C. Taylor, C. Wanke, D. Ramamurthy, J. Xie, 
%       "Multivariate Probabilistic Collocation Method for Effective Uncertainty 
%       Evaluation with Application to Air Traffic Management", IEEE Transactions 
%       on Systems, Man and Cybernetics: System, Vol. 44, No. 10, pp. 1347-1363, 2014.
%
%       Y. Zhou, D. Ramamurthy, Y. Wan, C. Taylor, and C. Wanke, "Multivariate
%       probabilistic collocation method for effective uncertainty evaluation
%       with application to air traffic management," in Proc. Amer. Control Conf.,
%       Jun. 2013, pp. 6345?6350.
%
%%% Date: 01/02/2019


function [g, meanOutput] = MPCM(symbols, distributions, ranges, degrees, G)
n = length(symbols); % number of parameters
f_joint = 1;
for i = 1:n
    f_joint = f_joint*distributions{i};
end
% find M-PCM points
SingleVariablePCM = cell(n,1);
Px = cell(n,1);
for i = 1:n
    variable = symbols(i);
    [SingleVariablePCM{i}, Px{i}] = PCMpoints(ranges(i,:), degrees(i), distributions{i},variable);
end

MPCMpoints = allCombinations(SingleVariablePCM);
NoofMPCMpoints = length(MPCMpoints(:,1)); % number of MPCM points

% Using M-PCM to construct low-order mapping and calculate the mean output 
OutputValues = zeros(NoofMPCMpoints,1);
for j = 1:NoofMPCMpoints
    OValue = subs(G, symbols(1), MPCMpoints(j,1));
    for i = 2:n
        OValue = subs(OValue, symbols(i), MPCMpoints(j,i));
    end
    OutputValues(j) = double(OValue);
end
% Calculate the L matrix
Term = OrthogonalTerms(Px);
% construct L matrix 
L = zeros(NoofMPCMpoints, length(Term));
D = zeros(NoofMPCMpoints, 1);
for j = 1:NoofMPCMpoints
    LValue = subs(Term, symbols(1), MPCMpoints(j,1));
    for i = 2:n
        LValue = subs(LValue, symbols(i), MPCMpoints(j,i));
    end
    L(j,:) = double(LValue);
    D(j,:) = 1/norm(L(j,:));
end
% Calculate the coefficients
D = diag(D);
B = (D*L)'*D*OutputValues;
% Calculate the formula of the low-order mapping g*
g = expand(vpa(sum(B'.*Term)));   
% Calculate mean output
meanOutput = B(1);
end


%=========== function: PCMpoints() ==============%
function [roots, Px] = PCMpoints(range, degree, f, variable)
% a = parameter(1);
% b = parameter(2);
% range = [a, b];
% f = distribution;
syms Px px;
Px(1)=0;
Px(2)=1;
px(1)=0;
px(2)=1;
if (degree>0)
    for i=3:(degree-1+3)
        A=sqrt(double(int(px(i-1)^2*f,range(1),range(2))));
        px(i)=variable*Px(i-1)-double(int(variable*Px(i-1)^2*f,range(1),range(2)))*Px(i-1)-A*Px(i-2);
        Px(i)=px(i)/sqrt(double(int(px(i)^2*f,range(1),range(2))));
    end
end
roots=double(solve(Px(i)));
end


%======== function: allCombinations() ==========%
function Allcombinations = allCombinations(points)
n = length(points(:,1));
% enter arguments backwards
ii = n:-1:1;
args = points;
% flip using ii if last column is changing fastest
[A{ii}] = ndgrid(args{ii}) ;
Allcombinations = reshape(cat(n+1,A{:}),[],n) ;
end


%=========== function: OrthogonalTerms() ===========%
function Term = OrthogonalTerms(Px)
n = length(Px);
% enter arguments backwards
ii = n:-1:1;
args = cell(1,n);
for i = 1:n
    args{i} = Px{i}(2:end-1);
end
% flip using ii if last column is changing fastest
[A{ii}] = ndgrid(args{ii}) ;
% concatenate
A = reshape(cat(n+1,A{:}),[],n) ;
Term = prod(A,2);
Term = reshape(Term,1,[]);
end



