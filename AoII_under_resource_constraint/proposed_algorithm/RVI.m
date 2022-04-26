%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file implements the Relative Value Iteration (RVI) algorithm for
% findind the optimal policy for the Markov Decision Process with
% truncated state space.
% See Algorithm 1 of the paper
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   N - Number of states
%   bound - Penalty constraints (Minimum possible penalty for each d)
%   lambda - Lagrange multiplier
%   smax - Truncation parameter (Maximum penalty)
%   epsilon - Tolerance (RVI precision)
%   P_trans - Transition probability matrix when transmission happens
%   P_notrans - Transition probability matrix when no transmission happens
% Output:
%   n - Optimal thresholds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function n = RVI(N, bound, lambda, smax, epsilon, P_trans, P_notrans)
    total = (N-1) * smax;  % Total number of value functions considered
    ref = 2;               % Index of reference state (randomly chosen)
    cnt = 0;               % Counter
    iter_max = 10000;      % Maximum iteration
    
    RV_old = zeros(total+1,1); % Relative value function matrix
    n = ones(N,1);             % Resulting threshold vector

    % Instant cost with/without transmission
    C_notrans = zeros(total+1,1);
    C_trans = zeros(total+1,1);
    for i = 1:N-1
        first = count(i,bound(i+1),smax);
        last = count(i,smax,smax);
        C_notrans(first:last) = (bound(i+1):smax);
        C_trans(first:last) = C_notrans(first:last) + lambda;
    end

    % Main Loop
    while cnt <= iter_max
        % Update value function
        Trans = P_trans * RV_old + C_trans;
        NoTrans = P_notrans * RV_old + C_notrans;
        tmp = min(Trans,NoTrans);
        RV_new = tmp - tmp(ref);
        
        % Reset the states that violates the constraints (Necessary)
        for i = 1:N-1
            first = count(i,1,smax);
            last = count(i,bound(i+1)-1,smax);
            RV_new(first:last) = 0;
        end

        % Generate the threshold vector
        if max(abs(RV_old - RV_new)) <= epsilon
            delta_V = Trans - NoTrans;
            Temp = reshape(delta_V(2:end),[smax,N-1]);
            for i = 1:N-1
                idx = find(Temp(:,i) < 0, 1);
                if idx
                    n(i+1) = idx;
                else
                    n(i+1) = smax;
                end
            end
            break
        end
        % One step forward
        RV_old = RV_new;
        cnt = cnt + 1;
    end

    % Reformate the resulting threshold vector (downward)
    for i = 2:N
       if n(i) <= bound(i)
          n(i) = 1;
       end
    end
end

%% function - count
% Return the the indices of state (d,s) in the matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   d - difference
%   s - penalty
%   smax - truncation parameter
% Output:
%   idx - the row/column index of state (d,s) in the matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function idx = count(d,s,smax)
    idx = 1 + (d - 1) * smax + s;
end
