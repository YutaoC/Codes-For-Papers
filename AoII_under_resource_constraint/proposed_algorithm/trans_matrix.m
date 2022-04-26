%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function constructs the transition probability matrix used in RVI
% algorithm.
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   N - Number of states
%   P - Probability of changing value
%   Ps - Probability of successful transmission
%   bound - Penalty constraints (Minimum possible penalty for each d)
%   smax - Truncation parameter (Maximum penalty)
% Output:
%   P_trans - Transition probability matrix when transmission happens
%   P_notrans - Transition probability matrix when no transmission happens
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P_trans,P_notrans] = trans_matrix(N,P,Ps,bound,smax)
    total = (N-1) * smax;  % Matrix dimension
    
    % Transition probabilities when no transmission happens
    P_notrans = zeros(total+1,total+1);
    % d = 0
    P_notrans(1,1) = 1 - 2 * P;
    P_notrans(1,2) = 2 * P;
    % d = 1
    for s = 1:smax
       row = count(1,s,smax);
       P_notrans(row,1) = P;
       P_notrans(row,count(1,min(smax,s+1),smax)) = 1 - 2 * P;
       P_notrans(row,count(2,min(smax,s+2),smax)) = P;
    end
    % 2 <= d <= N-2
    for d = 2:N-2
       for s = bound(d+1):smax
          row = count(d,s,smax);
          P_notrans(row,count(d-1,min(smax,s+d-1),smax)) = P;
          P_notrans(row,count(d,min(smax,s+d),smax)) = 1 - 2 * P;
          P_notrans(row,count(d+1,min(smax,s+d+1),smax)) = P;
       end
    end
    % d = N-1
    for s = bound(N):smax
       row = count(N-1,s,smax);
       P_notrans(row,count(N-1,min(smax,s+N-1),smax)) = 1 - 2 * P;
       P_notrans(row,count(N-2,min(smax,s+N-2),smax)) = 2 * P;
    end

    % Transition probabilities when transmission happens
    P_trans = (1 - Ps) * P_notrans;
    % d = 0
    P_trans(1,1) = P_trans(1,1) + Ps * (1 - 2 * P);
    P_trans(1,2) = P_trans(1,2) + 2 * Ps * P;
    % d = 1
    for s = 1:smax
       row = count(1,s,smax);
       P_trans(row,1) = P_trans(row,1) + Ps * (1 - 2 * P);
       P_trans(row,2) = P_trans(row,2) + 2 * Ps * P;
    end
    % 2 <= d <= N-2
    for d = 2:N-2
       for s = bound(d+1):smax
          row = count(d,s,smax);
          P_trans(row,1) = Ps * (1 - 2 * P);
          P_trans(row,2) = P_trans(row,2) + 2 * Ps * P;
       end
    end
    % d = N-1
    for s = bound(N):smax
       row = count(N-1,s,smax);
       P_trans(row,1) = Ps * (1 - 2 * P);
       P_trans(row,2) = P_trans(row,2) + 2 * Ps * P;
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
