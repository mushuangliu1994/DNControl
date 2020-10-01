%%% MPCMOFFD: a scalable uncertainty evaluation method that integrates
%             M-PCM and OFFD.
%   [g, meanOutput] = MPCMOFFD(symbols, distributions, range, tau, G)
%   estimates the mean output of a system with dynamics described by G. 
%
%%% The inputs of the MPCMOFFD function include:
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
%       tau: maximal cross-product power. Cross-terms of more than tau parameters 
%            in the system mapping are assumed to be negligible. 
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
%       tau = 2;
%       G = 1.5 - 0.5*x1^2  - 2*x3^3 + 5*x1^3*x2 + x4+x5^2;
%       [g, meanOutput] = MPCMOFFD(symbols, distributions, ranges, tau, G)
%
%%% References:
%       J. Xie, Y. Wan, K. Mills, J. J. Filliben, Y. Lei, Z. Lin, 
%       "M-PCM-OFFD: An effective output statistics estimation method for 
%       systems of high dimensional uncertainties subject to low-order 
%       parameter in-teractions", Mathematics and Computers in Simulation, 
%       https://doi.org/10.1016/j.matcom.2018.10.010, 2018 (in press).
%
%       J. Xie, Y. Wan, Y. Zhou, K. Mills, J.J. Filliben, and Y. Lei, 
%       "Effective and Scalable Uncertainty Evaluation for Large-Scale 
%       Complex System Applications", in Proceedings of Winter Simulation 
%       Conference (WSC), Savannah, GA, Dec. 2014.
%
%%% Date: 01/02/2019



function [g, meanOutput] = MPCMOFFD(symbols, distributions, ranges, tau, G)
n = length(symbols); % number of parameters
f_joint = 1;
for i = 1:n
    f_joint = f_joint*distributions{i};
end
degrees = 2*ones(1, n);
SingleVariablePCM = cell(n,1);
Px = cell(n,1);
for i = 1:n
    variable = symbols(i);
    [SingleVariablePCM{i}, Px{i}] = PCMpoints(ranges(i,:), degrees(i), distributions{i},variable);
end

% Find M-PCM-OFFD points
MPCMOFFDpoints = OFFDpoints(SingleVariablePCM, n, tau);
Noofpoints = length(MPCMOFFDpoints(:,1)); % number of MPCMOFFD points
% Using MPCMOFFD to construct low-order mapping and calculate the mean output
OutputValues = zeros(Noofpoints,1);
for j = 1:Noofpoints
    OValue = subs(G, symbols(1), MPCMOFFDpoints(j,1));
    for i = 2:n
        OValue = subs(OValue, symbols(i), MPCMOFFDpoints(j,i));
    end
    OutputValues(j) = double(OValue);
end
Term = PartOrthTerms(Px, tau);
% construct L matrix 
L = zeros(Noofpoints, length(Term));
D = zeros(Noofpoints, 1);
for j = 1:Noofpoints
    LValue = subs(Term, symbols(1), MPCMOFFDpoints(j,1));
    for i = 2:n
        LValue = subs(LValue, symbols(i), MPCMOFFDpoints(j,i));
    end
    L(j,:) = double(LValue);
    D(j,:) = 1/norm(L(j,:));
end
D = diag(D);
% Calculate the coefficients
B = (L'*L)^(-1)*L'*OutputValues;
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


%=========== function: OFFDpoints() =============%
function MPCMOFFDpoints = OFFDpoints(SingleVariablePCM, n, tau)
% total possible letters is 52
letters = 'a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z';
parameters = letters(1:2*n-1);
%% determine k and R
% k: reduce points to 2^k
% R: resolution
if n < 3 || tau > ceil(n/2)-1
    disp('Cannot use OFFD to further reduce the simulation points.');
    % cannot further reduce the points, find MPCM points
    dFF2 = ff2n(n);
    IndexMatrix_full = dFF2 + 1;
    MPCMOFFDpoints = ConstructCombinations(IndexMatrix_full, SingleVariablePCM);
else
    v = 0;
    for i = 1:tau
        v = v + nchoosek(n, i);
    end
    upbound = n - ceil(log2(v));
    for j = upbound:-1:1
        disp('Searching for generators for the OFFD.');
        k = n - j;
        R = 2*tau + 1;
        [generators, ~, notfound]=fracfactgenT(parameters,k,R);
        if ~notfound 
            break;
        end
    end
    [dfF,~] = fracfact(generators);
    IndexMatrix_offd = 0.5.*dfF + 1.5;
    MPCMOFFDpoints = ConstructCombinations(IndexMatrix_offd, SingleVariablePCM);
end
end

%========== function: ConstructCombinations() ==============%
function Points = ConstructCombinations(IndexMatrix, SingleVariablePCM)
Points = IndexMatrix;
Columns = length(IndexMatrix(1,:));
for i = 1:Columns
    roots = SingleVariablePCM{i};
    Points(:,i) = roots(IndexMatrix(:,i)); 
end
end


%=========== function: PartOrthTerms() ============%
function Term = PartOrthTerms(Px, tau)
n = length(Px);
% enter arguments backwards, so last one (AN) is changing fastest
ii = n:-1:1;
args = cell(1,n);
for i = 1:n
    args{i} = Px{i}(2:end-1);
end
% flip using ii if last column is changing fastest
[A{ii}] = ndgrid(args{ii});
% concatenate
A = reshape(cat(n+1,A{:}),[],n);
if tau < n
    LA = length(A(:,1));
    Delete = [];
    for i = 1:LA
        [~,c] = find(A(i,:) == 1);
        if length(c) < n-tau
            Delete = [Delete, i];
        end
    end
    A(Delete,:) = [];
end
Term = prod(A,2);
Term = reshape(Term,1,[]);
end


%=========== function: fracfactgenT()============%
function [gens,resfound, notfound] = fracfactgenT(effects,k,resneeded,basic,dispopt,doall)
% This function is modified based on the factfactgen() function

%FRACFACTGEN Fractional factorial design generators.
%   GENS = FRACFACTGEN(MODEL,K) finds a set of fractional factorial design
%   generators suitable for fitting a specified model.  MODEL specifies the
%   model, and is either a text string or a matrix of 0's and 1's as
%   accepted as the MODEL argument by the X2FX function.  The design will
%   have 2^K runs.  The output GENS is a cell array that specifies the
%   confounding of the design, and that is suitable for use as input to the
%   FRACFACT function. The FRACFACT function can generate the design and
%   display the confounding pattern for the generators.  If K is not given,
%   FRACFACTGEN will try to find the smallest possible value.
%
%   If MODEL is a text string, then MODEL must consist of a sequence of
%   words separated by spaces, each word representing a term that must be
%   estimable in the design.  The jth letter of the alphabet represents the
%   jth factor.  For example, 'a b c d ac' defines a model that includes
%   the main effects for factors a-d, and the interaction between factors a
%   and c. Use a-z for the first 26 factors, and if necessary A-Z for the
%   remaining factors.
%
%   FRACFACTGEN uses the Franklin-Bailey algorithm to find the generators
%   of a design that is capable of fitting the specified model.  MODEL must
%   not specify more than 52 factors.
%
%   GENS = FRACFACTGEN(MODEL,K,RES) tries to find a design with resolution
%   RES (default 3).  If FRACFACTGEN is unable to find the requested
%   resolution, it will either display an error, or if it located a
%   lower-resolution design capable of fitting the model, it will return
%   the generators for that design along with a warning.  If the result is
%   an error, it may still be possible to call FRACFACTGEN with a lower
%   value of RES and find a set of design generators.
%
%   GENS = FRACFACTGEN(MODEL,K,RES,BASIC) also accepts a vector BASIC with
%   K elements specifying the numbers of the factors that are to be treated
%   as basic factors.  These factors will receive single-letter generators,
%   and other factors will be confounded with interactions among the basic
%   factors.  The default is chosen to include factors that are part of the
%   highest-order interaction in MODEL.
%
%   Example:  Find the generators for a design with four factors and 2^3=8
%             runs so that we can estimate the interaction between the
%             first and third factors.
%
%       fracfactgen('a b c d ac',3)
%
%       m = [1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1;1 0 1 0];
%       fracfactgen(m,3)
%
%   See also FRACFACT.

% The following undocumented arguments are not supported and may change
% in a future release:
%   GENS = FRACFACTGEN(MODEL,K,RES,BASIC,DISPOPT) displays diagnostic output if
%   DISPOPT is true or omits it if DISPOPT is false (default).
%
%   GENS = FRACFACTGEN(MODEL,K,RES,BASIC,DISPOPT,DOALL) finds all possible
%   generators if DOALL is true, or only one set if DISPOPT is false (default).
%
%   [GENS,RESFOUND] = FRACFACTGEN(...) returns the resolution of the design
%   produced by these generators.  However, if the requested resolution is
%   3 (indicating that confounding with interactions is acceptable), the
%   RESFOUND value is set to 3 and the FRACFACTGEN function does not
%   attempt to determine if the resolution of the output generators is
%   actually higher than 3.

% Reference:
%   Franklin, M.F., and R. A. Bailey (1977), "Selection of defining
%   contrasts and confounded effects in two-level experiments," Applied
%   Statistics, 26, pp. 321-326. 

%   Copyright 2005-2012 The MathWorks, Inc.


% On input model may be a string such as 'a b c abc', or a matrix as
% accepted by x2fx.  Convert to a model matrix and an effects vector.
notfound = false;
alpha = [('a':'z'),('A':'Z')];
if ischar(effects)
    % For backward compatibility, if no a-z values are present, convert to
    % lower case
    if ~any(ismember(alpha(1:26),effects))
        effects = lower(effects);
    end
    effchars = effects(effects~=' ');
    [ok,loc] = ismember(effchars,alpha);
    if ~all(ok)
        error(message('stats:fracfactgen:BadModelString'));
    end
    
    n = max(loc);
    effects = textscan(effects,'%s');
    effects = effects{1};
    
    model = zeros(length(effects),n);
    for j=1:length(effects)
        model(j,:) = ismember(alpha(1:n),effects{j});
    end
elseif isnumeric(effects) && all(ismember(effects(:),0:1))
    model = effects;
else
    error(message('stats:fracfactgen:BadModel'));
end

% Fix up model by adding missing main effects and removing the constant
t = all(model==0,2);  % remove constant term, if any
model(t,:) = [];
n = size(model,2);
if n>52
    error(message('stats:fracfactgen:TooManyFactors'));
end
eyen = eye(n);
found = ismember(eyen,model,'rows');
if any(~found)
    model = [model;eyen(~found,:)];
end

effects = model * (2.^(0:n-1))';
if nargin<2 || isempty(k)
    k = [];
elseif(~isscalar(k) || ~isnumeric(k) || k~=round(k) || k<2)
    error(message('stats:fracfactgen:BadK'))
end

nterms = length(effects);

% Defaults for optional arguments
if nargin<3 || isempty(resneeded)
    resneeded = 3;
elseif ~isscalar(resneeded) || resneeded~=floor(resneeded) || resneeded<3
    error(message('stats:fracfactgen:BadRes'));
end
if nargin<4
    basic = [];
end
if nargin<5 || isempty(dispopt)
    dispopt = false;
elseif ~isequal(dispopt,true) && ~isequal(dispopt,false)
    error(message('stats:fracfactgen:BadDispopt'));
end
if nargin<6 || isempty(doall)
    doall = false;
elseif ~isequal(doall,true) && ~isequal(doall,false)
    error(message('stats:fracfactgen:BadDoall'));
end

if dispopt
    fprintf('%s\n',getString(message('stats:fracfactgen:InputEffects')));
    fprintf('%s\n',showterms(effects,alpha));
end

% Step 1.  Define ineligible effects set, ies
ies = zeros(1,nterms*(nterms+1)/2);
ies(1:nterms) = effects;
base = nterms;

for i=1:nterms-1
    j = nterms-i;
    ies(base+(1:j)) = bitor(effects(i),effects(i+1:end));
    base = base+j;
end

ies = sort(ies);
ies(diff(ies)==0) = [];
if dispopt
    fprintf('%s\n',getString(message('stats:fracfactgen:IneligibleEffects')));
    fprintf('%s\n',showterms(ies,alpha));
end

% Step 2.  Choose smallest possible sample size
termsize = bitcount(effects);
maxtermsize = max(termsize);
kmin = max(ceil(log2(nterms+1)),maxtermsize);
if resneeded>3
    if resneeded==4
        % For res 4, must be at least as large as a res 3 foldover
        kmin = max(kmin, ceil(log2(2*n)));
    else
        % For res 5, must be able to fit all terms up to 2-factor interactions
        kmin = max(kmin, ceil(log2(1+n+n*(n-1)/2)));
    end
end
if isempty(k)
    k = kmin;
elseif k<kmin
    notfound = true;
%     error(message('stats:fracfactgen:KTooSmall', kmin));
end
m = n - k;
if m <= 0
    warning(message('stats:fracfactgen:NotFractional'));
    gens = cellstr(char('a'+(0:min(k,n)-1)'));
    resfound = 5;
    return
end

% Step 3.  Select basic factors
if ~isempty(basic)
    basic = sort(basic(:));
    if any(~ismember(basic,1:n)) || any(diff(basic)==0) || length(basic)~=k
        error(message('stats:fracfactgen:BadBasic', k, n));
    end
else
    % Start with one of the largest terms (highest level of interaction)
    inter = find(termsize==maxtermsize,1);
    basic = zeros(1,n);
    for j=1:n
        if bitget(effects(inter),j)
            basic(j) = 1;
        end
    end
    j = 1;
    nbasic = sum(basic);
    while(nbasic<k && j<n)        % select more factors if more are needed
        if ~basic(j)
            basic(j) = 1;
            nbasic = nbasic+1;
        end
        j = j+1;
    end
    basic = find(basic);
end
added = find(~ismember(1:n,basic));    % get added (non-basic) factors

if dispopt
    fprintf('%s\n',getString(message('stats:fracfactgen:BasicFactors')));
    fprintf('%s ',alpha(basic));
    fprintf('\n');
end

% Form basic effects group (all products of them), omitting 0th and 1st
% order effects, as well as other effects depending on the required
% resolution.
indexbe = (3:2^k-1)';
indexbitcount = bitcount(indexbe);
minbitcount = resneeded-1;
indexbe(indexbitcount<minbitcount) = [];
basicgroup = zeros(numel(indexbe),1);
for j=1:k
    basicgroup = bitset(basicgroup,basic(j),bitget(indexbe,j));
end
basicgroup(ismember(basicgroup,effects)) = [];
addedterms = model(:,added);

% The rest of this step does not appear in the F&B paper.  First, determine
% if the model is symmetric.  Then we will reduce our search by requiring
% the row numbers in cursel to be decreasing, since we don't need to
% consider permutations of the cursel vectors we've already examined.
mainonly = all(sum(addedterms,1)<=1);
symmetric = mainonly && all(sum(addedterms,2)<=1);

% We don't need to form all identities if we need just resolution 3 and
% there are no added term interactions
skipidcheck = (resneeded<=3 && symmetric);
if ~skipidcheck && resneeded==4 && m>10 && mainonly
    % If m is large, try a trick to avoid allocating a big array.  If we
    % restrict the basic group to values with bit counts that are all even
    % or all odd, we insure resolution 4 without having to create the
    % identity relations, so long as there are no interactions among the
    % added factors.
    t = mod(bitcount(basicgroup),2)==1;
    if sum(t)<length(t)/2
        t = ~t;
    end
    basicgroup = basicgroup(t);
    skipidcheck = true;
end
if ~skipidcheck
    identities = zeros(2^m-1,1); % current defining contrasts
    bitcounts = zeros(2^m-1,1);  % number of bits in each defining contrast
end

% Step 4.  Create table of eligible effects
table = zeros(numel(basicgroup),m);
for j=1:m
    table(:,j) = bitset(basicgroup,added(j),1);
end
table(ismember(table,ies)) = 0;
t = all(table==0,2);
if any(t)
    table(t,:) = [];
end
nrows = size(table,1);

% Step 5.  Initialize for search through table
cursel = zeros(1,m);             % current selection of generators from table
col = 0;                         % current column
fwd = true;                      % moving forward through columns
resfound = 0;                    % best resolution found so far
gens = cell(n,1);                % generators
tablebc = bitcount(table);       % length of each potential generator

% Step 6.  Move to next column
while(~notfound && doall || resfound<resneeded)
    if fwd
        col = col+1;
        if symmetric && col>1 && ~doall
            cursel(col) = max(1, min(cursel(1:col-1)));
        else
            cursel(col) = nrows+1;
        end
    else
        col = col-1;
        fwd = true;
    end
    nident = 2^(col-1) - 1;
    while cursel(col)>1 && (doall || resfound<resneeded)

        % Step 7.  Select the next available effect in this column
        cursel(col) = cursel(col)-1;
        if dispopt>=2
            idx = sub2ind(size(table),cursel(1:col),1:col);
            disp(showterms(table(idx),alpha));
        end
        if any(cursel(1:col-1) == cursel(col))
            continue   % already selected
        end
        if ~doall && tablebc(cursel(col),col) <= resfound-1
            continue   % no better than generators already found
        end
        gen = table(cursel(col),col);
        if gen==0
            continue   % not eligible
        end

        % Step 8.  Make sure product of this with identities is eligible
        if ~skipidcheck
            identities(nident+1) = gen;
            for j=1:nident
                identities(nident+j+1) = bitxor(identities(j),gen);
            end
            newrows = nident + (1:nident+1);
            bitcounts(newrows) = bitcount(identities(newrows));
            if ~doall && min(bitcounts(newrows))<=resfound
                continue
            end
            if any(ismember(identities(nident+(1:nident+1)),ies))
                continue
            end
        end

        % Step 9.  Extend the defining contrasts group
        if col==m
            if ~skipidcheck
                resfound = min(bitcounts);
            else
                resfound = resneeded;
            end
            gens(basic) = cellstr(alpha(basic)');
            for j=1:length(added)
                gens{added(j)} = showterms(bitset(table(cursel(j),j),added(j),0),alpha);
            end

            if dispopt
                fprintf(getString(message('stats:fracfactgen:ResolutionFound',resfound)));
                fprintf('    ');
                for j=1:n
                    fprintf('%s ',gens{j});
                end
                fprintf('\n');
            end
            
            % May need to back up to beat the best resolution so far
            if ~doall && ~skipidcheck
                % Find the first col that fails to improve upon our best
                % design so far.  If the current design is already
                % adequate, we won't actually back up
                firstbad = find(bitcounts<=resfound,1,'first');
                badcol = 1 + ceil(sqrt(2*firstbad+.25) - .5 - 100*eps);
                if badcol<col
                    col = badcol;
                    fwd = false;
                    break
                end
            end
        else
            break
        end
    end

    if cursel(col) > 1
        continue
    end

    % Step 10.  Back to earlier column
    if col==1
        break
    end
    fwd = false;
end

if resfound==0
    notfound = true;
%     error(message('stats:fracfactgen:NoDesign', 2^k));
elseif resfound<resneeded
    warning(message('stats:fracfactgen:LowResolution', resneeded, resfound));
end

% ----------------------
function t=showterms(v,alpha)
% Represent terms in vector v as a character string
t = '';
for i=1:length(v)
    t = [t ' '];
    vi = v(i);
    j = 1;
    if vi==0
        t = [t '1'];
    else
        while(vi)
            if bitget(vi,j)
                t = [t alpha(j)];
                vi = bitset(vi,j,0);
            end
            j = j+1;
        end
    end
end
t = t(2:end);
end

% -----------------------
function c = bitcount(v)
% Count the number of bits set in each element of v
c = zeros(size(v));
t = find(v~=0);
j = 1;
while(~isempty(t))
    mask = bitget(v(t),j)==1;
    c(t(mask)) = c(t(mask))+1;
    v(t) = bitset(v(t),j,0);
    t = t(v(t)~=0);
    j = j+1;
end
end
end







