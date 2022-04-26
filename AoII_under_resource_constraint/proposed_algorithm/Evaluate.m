%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file calculates the expected AoII of the system and the expected
% transmission ratio of the transmitter resulting from the adoption of 
% "Threshold update policy" for given thresholds.
% See Proposition 2 and Corollary 2 of the paper
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   N - Number of states
%   P - Probability of changing value
%   Ps - Probability of successful transmission
%   n - Thresholds
%   bound - Penalty constraints (Minimum possible penalty for each d)
%   isAoII - Boolean indicates whether AoII is needed
%   bound - Minimum penalty constraints
% Output:
%   AoII - The expected Age of Incorrect Information
%   Ratio - The expected transmission ratio
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [AoII, Ratio] = Evaluate(N,P,Ps,n,bound,isAoII)
    % Obtain tau
    tau = n(2) + 2;
    for i = 2:N-1
       if n(i+1) + i + 1 > tau
           tau = n(i+1) + i + 1;
       end
    end
    
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
          idx = count(d,s,tau);
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
          idx = count(d,s,tau);
          A(2,idx) = 2 * Ps * P;
       end
    end
    for s = 2:tau-1
        idx1 = count(1,s-1,tau);
        idx2 = count(1,s,tau);
        idx3 = count(2,s-1,tau);
        A(idx2,idx1) = (1 - Ps * (s-1>=n(2))) * (1 - 2 * P);
        A(idx2,idx2) = -1;
        A(idx2,idx3) = (1 - Ps * (s-1>=n(3))) * P;
    end
    idx1 = count(1,tau,tau);
    idx2 = count(2,tau,tau);
    A(idx1,idx1-1) = Pf * (1 - 2 * P);
    A(idx1,idx1) = Pf * (1 - 2 * P) - 1;
    A(idx1,idx2-1) = Pf * P;
    A(idx1,idx2) = Pf * P;

    % d = 2 - d = N - 3
    for d = 2:N-3
        for s = 1:min(bound(d+1),tau)-1
            idx = count(d,s,tau);
            A(idx,idx) = 1;
        end
        for s = bound(d+1):tau-1
            idx1 = count(d-1,s-d,tau);
            idx2 = count(d,s-d,tau);
            idx3 = count(d,s,tau);
            idx4 = count(d+1,s-d,tau);
            A(idx3,idx1) = (1 - Ps * ((s-d)>=n(d))) * P;
            A(idx3,idx2) = (1 - Ps * ((s-d)>=n(d+1))) * (1 - 2 * P);
            A(idx3,idx3) = -1;
            A(idx3,idx4) = (1 - Ps * ((s-d)>=n(d+2))) * P;
        end
        row = count(d,tau,tau);
        for s = tau-d : tau
            idx1 = count(d-1,s,tau);
            idx2 = count(d,s,tau);
            idx3 = count(d+1,s,tau);
            A(row,idx1) = Pf * P;
            A(row,idx2) = Pf * (1 - 2 * P);
            A(row,idx3) = Pf * P;
        end
        A(row,row) = A(row,row) - 1;
    end

    % d = N - 2
    for s = 1:min(bound(N-1),tau)-1
        idx = count(N-2,s,tau);
        A(idx,idx) = 1;
    end
    for s = bound(N-1):tau-1
        idx1 = count(N-3,s-N+2,tau);
        idx2 = count(N-2,s-N+2,tau);
        idx3 = count(N-2,s,tau);
        idx4 = count(N-1,s-N+2,tau);
        A(idx3,idx1) = (1 - Ps * ((s-N+2)>=n(N-2))) * P;
        A(idx3,idx2) = (1 - Ps * ((s-N+2)>=n(N-1))) * (1 - 2 * P);
        A(idx3,idx3) = -1;
        A(idx3,idx4) = 2 * (1 - Ps * ((s-N+2)>=n(N))) * P;
    end
    row = count(N-2,tau,tau);
    for s = tau-N+2 : tau
        idx1 = count(N-3,s,tau);
        idx2 = count(N-2,s,tau);
        idx3 = count(N-1,s,tau);
        A(row,idx1) = Pf * P;
        A(row,idx2) = Pf * (1 - 2 * P);
        A(row,idx3) = 2 * Pf * P;
    end
    A(row,row) = A(row,row) - 1;

    % d = N - 1
    for s = 1:min(bound(N),tau)-1
        idx = count(N-1,s,tau);
        A(idx,idx) = 1;
    end
    for s = bound(N):tau-1
        idx1 = count(N-2,s-N+1,tau);
        idx2 = count(N-1,s-N+1,tau);
        idx3 = count(N-1,s,tau);
        A(idx3,idx1) = (1 - Ps * ((s-N+1)>=n(N-1))) * P;
        A(idx3,idx2) = (1 - Ps * ((s-N+1)>=n(N))) * (1 - 2 * P);
        A(idx3,idx3) = -1;
    end
    row = count(N-1,tau,tau);
    for s = tau-N+1 : tau
        idx1 = count(N-2,s,tau);
        idx2 = count(N-1,s,tau);
        A(row,idx1) = Pf * P;
        A(row,idx2) = Pf * (1 - 2 * P);
    end
    A(row,row) = A(row,row) - 1;

    % Last Row
    A(M+2,:) = ones(1,M+1);

    % vector b
    b(end) = 1;

    % Solve for the stationary distribution
    dist = linsolve(A(2:end,:),b(2:end));

    % For the expected transmission ratio
    x = zeros(M+1,1);
    for d = 1:N-1
       for s = n(d+1):tau
           idx = count(d,s,tau);
           x(idx) = 1;
       end
    end
    Ratio = sum(dist .* x);
    
    % For the expected AoII
    if isAoII
        E = zeros(M+1, M+1);
        E(1,1) = 1;
        E(2,2) = -1;
        E(3:tau+1,:) = A(3:tau+1,:);
        E(tau+2:end,:) = A(tau+2:end-1,:);
        k = zeros(M+1,1);
        for d = 1:N-1
           k((d-1)*tau+2:d*tau+1) = (-d) * ones(1,tau); 
        end
        f = dist .* k;
        AoII = sum(linsolve(E,f));
    else
        AoII = 0;
    end
end

%% function - count
% Return the the indices of state (d,s) in the matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   d - difference
%   s - penalty
%   tau - tau
% Output:
%   idx - the row/column index of state (d,s) in the matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function idx = count(d,s,tau)
    idx = 1 + (d - 1) * tau + s;
end
