%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file approximates the stationary distribution of the Markov chain
% induced from the "Threshold update policy" for given thresholds.
% (The reduction of complexity is obvious when thresholds are huge)
% See Corollary 1 of the paper (Not exact by equivalent)
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% System parameters
N = 7;     % Number of states in the Markovian source
P = 0.2;   % Probability of changing state
Ps = 0.8;  % Probability of successful transmission

n = [1,964,867,700,669,652,444]; % Thresholds

% Minimum penalty constraint for each difference
bound = zeros(1,N);
for d = 2:N
   bound(d) = bound(d-1) + d - 1; 
end

% tau
tau = n(2) + 2;
for i = 2:N-1
   if n(i+1) + i + 1 > tau
       tau = n(i+1) + i + 1;
   end
end

% Reformulate the threshold
n = up_format(n,bound);

% eta
eta = n(end) - 1;

%% Pre-calculation
% The stationary distribution Q resulting from thresholds [eta,...,eta]
n_pre = eta * ones(1,N);
n_pre(1) = 1;

tau_pre = n_pre(2) + 2;
for i = 2:N-1
   if n_pre(i+1) + i + 1 > tau_pre
       tau_pre = n_pre(i+1) + i + 1;
   end
end

dist = Theoretical(N,P,Ps,n_pre,tau_pre,bound);
Q = reshape(dist(2:end),[tau_pre,N-1]);

%% Approximated stationary distribution
dist_theo_approx = Theoretical_approx(N,P,Ps,n,tau,eta,Q);

%% Function - Theoretical_approx
% Calculate the stationary distribution (approximated)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   N - Number of states
%   P - Probability of changing value
%   Ps - Probability of successful transmission
%   n - Thresholds
%   tau - tau
%   eta - eta
%   Q - the stationary distribution used for approximation
% Output:
%   dist_theo_approx - The approximated stationary distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dist_theo_approx = Theoretical_approx(N,P,Ps,n,tau,eta,Q)
    Pf = 1 - Ps;              % Probability of failed transmission
    M = (N-1) * (tau-eta+1);  % Number of variables when d >= 1

    Q_eta = sum(Q(1:eta,:),1);  % Auxiliary

    % The system of linear equations for the stationary distribution
    A = zeros(M+2, M+2);  % Empty matrix A
    b = zeros(M+2,1);     % Empty vector b

    % d = 1
    A(2,1) = 2 * P;
    A(2,2) = A(2,2) -1;
    for d = 1:N-1
       for s = n(d+1):tau
          idx = count(d,s,tau,eta);
          A(2,idx) = A(2,idx) + 2 * Ps * P;
       end
    end
    A(3,2) = A(3,2) + 1;
    A(3,3) = A(3,3) - 2 * P;
    A(3,4) = A(3,4) - 1;
    idx = count(2,eta,tau,eta);
    A(3,idx) = A(3,idx) + P;
    for s = eta+1:eta+1
        row = count(1,s,tau,eta);
        idx1 = count(1,eta,tau,eta);
        idx2 = count(2,eta,tau,eta);
        A(row,idx1) = A(row,idx1) + Q(s-1,1)/Q_eta(1)*(1 - Ps * (s-1>=n(2))) * (1 - 2 * P);
        A(row,row) = A(row,row) - 1;
        A(row,idx2) = A(row,idx2) + Q(s-1,2)/Q_eta(2)*(1 - Ps * (s-1>=n(3))) * P;
    end
    for s = eta+2:tau-1
        idx1 = count(1,s,tau,eta);
        idx2 = count(2,s-1,tau,eta);
        A(idx1,idx1) = A(idx1,idx1) - 1;
        A(idx1,idx1-1) =  A(idx1,idx1-1) + (1 - Ps * (s-1>=n(2))) * (1 - 2 * P);
        A(idx1,idx2) = A(idx1,idx2) + (1 - Ps * (s-1>=n(3))) * P;
    end
    idx1 = count(1,tau,tau,eta);
    idx2 = count(2,tau,tau,eta);
    A(idx1,idx1-1) = A(idx1,idx1-1) + Pf * (1 - 2 * P);
    A(idx1,idx1) = A(idx1,idx1) + Pf * (1 - 2 * P) - 1;
    A(idx1,idx2-1) = A(idx1,idx2-1) + Pf * P;
    A(idx1,idx2) = A(idx1,idx2) + Pf * P;

    % d = 2 - d = N - 3 
    for d = 2:N-3
        idx1 = count(d,eta,tau,eta);
        idx2 = count(d-1,eta,tau,eta);
        idx3 = count(d+1,eta,tau,eta);
        A(idx1,idx2) = A(idx1,idx2) + P;
        A(idx1,idx1) = A(idx1,idx1) - 2 * P;
        A(idx1,idx3) = A(idx1,idx3) + P;
        for s = eta+1:eta+d
           idx = count(d,s,tau,eta);
           A(idx1,idx) = A(idx1,idx) - 1;
        end
        for s = eta + 1:eta + d
            row = count(d,s,tau,eta);
            idx1 = count(d-1,eta,tau,eta);
            idx2 = count(d,eta,tau,eta);
            idx3 = count(d+1,eta,tau,eta);
            A(row,idx1) = A(row,idx1) + Q(s-d,d-1)/Q_eta(d-1)*(1 - Ps * ((s-d)>=n(d))) * P;
            A(row,row) = A(row,row) - 1;
            A(row,idx2) = A(row,idx2) + Q(s-d,d)/Q_eta(d)*(1 - Ps * ((s-d)>=n(d+1))) * (1 - 2 * P);
            A(row,idx3) = A(row,idx3) + Q(s-d,d+1)/Q_eta(d+1)*(1 - Ps * ((s-d)>=n(d+2))) * P;
        end
        for s = eta + d + 1:tau-1
            idx1 = count(d-1,s-d,tau,eta);
            idx2 = count(d,s-d,tau,eta);
            idx3 = idx2 + d;
            idx4 = count(d+1,s-d,tau,eta);
            A(idx3,idx1) = A(idx3,idx1) + (1 - Ps * ((s-d)>=n(d))) * P;
            A(idx3,idx2) = A(idx3,idx2) + (1 - Ps * ((s-d)>=n(d+1))) * (1 - 2 * P);
            A(idx3,idx3) = A(idx3,idx3) - 1;
            A(idx3,idx4) = A(idx3,idx4) + (1 - Ps * ((s-d)>=n(d+2))) * P;
        end
        row = count(d,tau,tau,eta);
        for s = tau-d : tau
            idx1 = count(d-1,s,tau,eta);
            idx2 = count(d,s,tau,eta);
            idx3 = count(d+1,s,tau,eta);
            A(row,idx1) = A(row,idx1) + Pf * P;
            A(row,idx2) = A(row,idx2) + Pf * (1 - 2 * P);
            A(row,idx3) = A(row,idx3) + Pf * P;
        end
        A(row,row) = A(row,row) - 1;
    end

    % d = N - 2
    d = N-2;
    idx1 = count(d,eta,tau,eta);
    idx2 = count(d-1,eta,tau,eta);
    idx3 = count(d+1,eta,tau,eta);
    A(idx1,idx1) = A(idx1,idx1) - 2 * P;
    A(idx1,idx2) = A(idx1,idx2) + P;
    A(idx1,idx3) = A(idx1,idx3) + 2 * P;
    for s = eta+1:eta+d
       idx = count(d,s,tau,eta);
       A(idx1,idx) = A(idx1,idx) - 1;
    end
    for s = eta + 1:eta + d
        row = count(d,s,tau,eta);
        idx1 = count(d-1,eta,tau,eta);
        idx2 = count(d,eta,tau,eta);
        idx3 = count(d+1,eta,tau,eta);
        A(row,idx1) = A(row,idx1) + Q(s-d,d-1)/Q_eta(d-1)*(1 - Ps * ((s-d)>=n(d))) * P;
        A(row,row) = A(row,row) - 1;
        A(row,idx2) = A(row,idx2) + Q(s-d,d)/Q_eta(d)*(1 - Ps * ((s-d)>=n(d+1))) * (1 - 2 * P);
        A(row,idx3) = A(row,idx3) + Q(s-d,d+1)/Q_eta(d+1)*(1 - Ps * ((s-d)>=n(d+2))) * 2 * P;
    end
    for s = eta + d + 1:tau-1
        idx1 = count(d-1,s-d,tau,eta);
        idx2 = count(d,s-d,tau,eta);
        idx3 = idx2 + d;
        idx4 = count(d+1,s-d,tau,eta);
        A(idx3,idx1) = A(idx3,idx1) + (1 - Ps * ((s-d)>=n(d))) * P;
        A(idx3,idx2) = A(idx3,idx2) + (1 - Ps * ((s-d)>=n(d+1))) * (1 - 2 * P);
        A(idx3,idx3) = A(idx3,idx3) - 1;
        A(idx3,idx4) = A(idx3,idx4) + (1 - Ps * ((s-d)>=n(d+2))) * 2 * P;
    end
    row = count(d,tau,tau,eta);
    for s = tau-d : tau
        idx1 = count(d-1,s,tau,eta);
        idx2 = count(d,s,tau,eta);
        idx3 = count(d+1,s,tau,eta);
        A(row,idx1) = A(row,idx1) + Pf * P;
        A(row,idx2) = A(row,idx2) + Pf * (1 - 2 * P);
        A(row,idx3) = A(row,idx3) + Pf * 2 * P;
    end
    A(row,row) = A(row,row) - 1;

    % d = N - 1
    d = N-1;
    idx1 = count(d,eta,tau,eta);
    idx2 = count(d-1,eta,tau,eta);
    A(idx1,idx2) = A(idx1,idx2) + P;
    A(idx1,idx1) = A(idx1,idx1) - 2 * P;
    for s = eta + 1:eta+d
       idx = count(d,s,tau,eta);
       A(idx1,idx) = A(idx1,idx) - 1;
    end
    for s = eta + 1:eta + d
        row = count(d,s,tau,eta);
        idx1 = count(d-1,eta,tau,eta);
        idx2 = count(d,eta,tau,eta);
        A(row,idx1) = A(row,idx1) + Q(s-d,d-1)/Q_eta(d-1)*(1 - Ps * ((s-d)>=n(d))) * P;
        A(row,row) = A(row,row) - 1;
        A(row,idx2) = A(row,idx2) + Q(s-d,d)/Q_eta(d)*(1 - Ps * ((s-d)>=n(d+1))) * (1 - 2 * P);
    end
    for s = eta + d + 1:tau-1
        idx1 = count(d-1,s-d,tau,eta);
        idx2 = count(d,s-d,tau,eta);
        idx3 = idx2 + d;
        A(idx3,idx1) = A(idx3,idx1) + (1 - Ps * ((s-d)>=n(d))) * P;
        A(idx3,idx2) = A(idx3,idx2) + (1 - Ps * ((s-d)>=n(d+1))) * (1 - 2 * P);
        A(idx3,idx3) = A(idx3,idx3) - 1;
    end
    row = count(d,tau,tau,eta);
    for s = tau-d : tau
        idx1 = count(d-1,s,tau,eta);
        idx2 = count(d,s,tau,eta);
        A(row,idx1) = A(row,idx1) + Pf * P;
        A(row,idx2) = A(row,idx2) + Pf * (1 - 2 * P);
    end
    A(row,row) = A(row,row) - 1;

    % First Row
    A(1,:) = ones(1,M+2);
    A(1,2) = 0;

    % vector b
    b(1) = 1;

    % Solve for the stationary distribution
    dist_theo_approx = linsolve(A,b);

%     % For the expected transmission ratio (approximated)
%     x = zeros(M+1,1);
%     for d = 1:N-1
%        for s = n(d+1):tau
%            idx = count(d,s,tau,eta);
%            x(idx) = 1;
%        end
%     end
%     Ratio_approx = sum(dist .* x);
end

%% Function - Theoretical
% Calculate the stationary distribution (exact)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   N - Number of states
%   P - Probability of changing value
%   Ps - Probability of successful transmission
%   n - Thresholds
%   tau - tau
%   bound - Minimum penalty constraints
% Output:
%   dist_theo - The stationary distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dist_theo = Theoretical(N,P,Ps,n,tau,bound)
    Pf = 1 - Ps;      % Probability of failed transmission
    M = (N-1) * tau;  % Number of variables when d >= 1

    % The system of linear equations for the stationary distribution
    A = zeros(M+2, M+1);  % Empty matrix A
    b = zeros(M+2,1);     % Empty vector b

    % d = 0
    A(1,1) = -2 * P;
    for s = 1:n(2)-1
       A(1,s+1) = P; 
    end
    for s = n(2):tau
       A(1,s+1) = Pf * P + Ps * (1 - 2 * P);
    end
    for d = 2:N-1
       for s = n(d+1):tau
          idx = count1(d,s,tau);
          A(1,idx) = Ps * (1 - 2 * P);
       end
    end
    
    % d = 1
    A(2,1) = 2 * P;
    for s = n(2):tau
        A(2,s+1) = 2 * Ps * P;
    end
    A(2,2) = A(2,2) -1;
    for d = 2:N-1
       for s = n(d+1):tau
          idx = count1(d,s,tau);
          A(2,idx) = 2 * Ps * P;
       end
    end
    for s = 2:tau-1
        idx1 = count1(1,s-1,tau);
        idx2 = count1(1,s,tau);
        idx3 = count1(2,s-1,tau);
        A(idx2,idx1) = (1 - Ps * (s-1>=n(2))) * (1 - 2 * P);
        A(idx2,idx2) = -1;
        A(idx2,idx3) = (1 - Ps * (s-1>=n(3))) * P;
    end
    idx1 = count1(1,tau,tau);
    idx2 = count1(2,tau,tau);
    A(idx1,idx1-1) = Pf * (1 - 2 * P);
    A(idx1,idx1) = Pf * (1 - 2 * P) - 1;
    A(idx1,idx2-1) = Pf * P;
    A(idx1,idx2) = Pf * P;

    % d = 2 - d = N - 3  
    for d = 2:N-3
        for s = 1:min(bound(d+1),tau)-1
            idx = count1(d,s,tau);
            A(idx,idx) = 1;
        end
        for s = bound(d+1):tau-1
            idx1 = count1(d-1,s-d,tau);
            idx2 = count1(d,s-d,tau);
            idx3 = count1(d,s,tau);
            idx4 = count1(d+1,s-d,tau);
            A(idx3,idx1) = (1 - Ps * ((s-d)>=n(d))) * P;
            A(idx3,idx2) = (1 - Ps * ((s-d)>=n(d+1))) * (1 - 2 * P);
            A(idx3,idx3) = -1;
            A(idx3,idx4) = (1 - Ps * ((s-d)>=n(d+2))) * P;
        end
        row = count1(d,tau,tau);
        for s = tau-d : tau
            idx1 = count1(d-1,s,tau);
            idx2 = count1(d,s,tau);
            idx3 = count1(d+1,s,tau);
            A(row,idx1) = Pf * P;
            A(row,idx2) = Pf * (1 - 2 * P);
            A(row,idx3) = Pf * P;
        end
        A(row,row) = A(row,row) - 1;
    end

    % d = N - 2
    for s = 1:min(bound(N-1),tau)-1
        idx = count1(N-2,s,tau);
        A(idx,idx) = 1;
    end
    for s = bound(N-1):tau-1
        idx1 = count1(N-3,s-N+2,tau);
        idx2 = count1(N-2,s-N+2,tau);
        idx3 = count1(N-2,s,tau);
        idx4 = count1(N-1,s-N+2,tau);
        A(idx3,idx1) = (1 - Ps * ((s-N+2)>=n(N-2))) * P;
        A(idx3,idx2) = (1 - Ps * ((s-N+2)>=n(N-1))) * (1 - 2 * P);
        A(idx3,idx3) = -1;
        A(idx3,idx4) = 2 * (1 - Ps * ((s-N+2)>=n(N))) * P;
    end
    row = count1(N-2,tau,tau);
    for s = tau-N+2 : tau
        idx1 = count1(N-3,s,tau);
        idx2 = count1(N-2,s,tau);
        idx3 = count1(N-1,s,tau);
        A(row,idx1) = Pf * P;
        A(row,idx2) = Pf * (1 - 2 * P);
        A(row,idx3) = 2 * Pf * P;
    end
    A(row,row) = A(row,row) - 1;

    % d = N - 1
    for s = 1:min(bound(N),tau)-1
        idx = count1(N-1,s,tau);
        A(idx,idx) = 1;
    end
    for s = bound(N):tau-1
        idx1 = count1(N-2,s-N+1,tau);
        idx2 = count1(N-1,s-N+1,tau);
        idx3 = count1(N-1,s,tau);
        A(idx3,idx1) = (1 - Ps * ((s-N+1)>=n(N-1))) * P;
        A(idx3,idx2) = (1 - Ps * ((s-N+1)>=n(N))) * (1 - 2 * P);
        A(idx3,idx3) = -1;
    end
    row = count1(N-1,tau,tau);
    for s = tau-N+1 : tau
        idx1 = count1(N-2,s,tau);
        idx2 = count1(N-1,s,tau);
        A(row,idx1) = Pf * P;
        A(row,idx2) = Pf * (1 - 2 * P);
    end
    A(row,row) = A(row,row) - 1;

    % Last Row (probabilities add up to 1)
    A(M+2,:) = ones(1,M+1);

    % vector b
    b(end) = 1;

    % Solve for the stationary distribution
    dist_theo = linsolve(A(2:end,:),b(2:end));
end

%% Function - count
% Return the index of state (d,s) in the matrix for the given data (approx)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   d - difference
%   s - penalty
%   tau - tau
%   eta - eta
% Output:
%   idx - the row/column index of state (d,s) in the matrix (approx)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function idx = count(d,s,tau,eta)
    idx = 2 + (d - 1) * (tau-eta+1) + s - eta + 1;
end

%% Function - count1
% Return the index of state (d,s) in the matrix for the given data (exact)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   d - difference
%   s - penalty
%   tau - tau
% Output:
%   idx - the row/column index of state (d,s) in the matrix (exact)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function idx = count1(d,s,tau)
    idx = 1 + (d - 1) * tau + s;
end

%% Function - up_format
% Return the largest thresholds among the equivalent thresholds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   n - thresholds
%   bound - bound
% Output:
%   ret - the resulting thresholds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ret = up_format(n,bound)
    ret = ones(1,length(n));
    for i = 2:length(n)
        if n(i) <= bound(i)
            ret(i) = min(ret(i-1),bound(i));
        else
            ret(i) = n(i);
        end
    end
end
