%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file verifies Condition 1 in the paper when the transmission time
% for an update is deterministic under various system parameters.
% See Section V.A of the paper
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% Store result
tmax_range = 15;
value_intervals = 20;
sigma = zeros(value_intervals,tmax_range);
delta_0 = zeros(value_intervals,tmax_range);
delta_1 = zeros(value_intervals,tmax_range);
con1 = zeros(value_intervals,tmax_range);
con2 = zeros(value_intervals,tmax_range);

%% Main Loop
for tmax = 1 : tmax_range
    for i = 2: value_intervals + 1
        p = (i - 1) * 0.05 / 2;
        P_chain = [(1-p),p;p,(1-p)];

        % Instant Costs
        C = zeros(100+1,1);
        for j = 0:100
            C(j+1) = InstantCost(j,tmax,p,P_chain);
        end

        % State Transition Probabilities
        P = zeros(2,tmax+1);
        tmp = P_chain^tmax;
        P(1,1) = tmp(1,1);
        P(2,1) = tmp(1,1);
        for s = 2:tmax+1
           tmp = P_chain^(tmax-s+1);
           P(1,s) = tmp(1,1) * p * (1-p)^(s-2);
        end
        tmp = P_chain^(tmax-1);
        P(2,2) = tmp(1,2) * (1-p);
        P(2,end) = p * (1-p)^(tmax-1);
        for s = 3:tmax
           tmp = P_chain^(tmax-s+1);
           P(2,s) = tmp(1,2) * p^2 * (1-p)^(s-3);
        end
        
        % Auxiliary quantities
        sigma(i,tmax) = Cal_sigma(p, tmax);
        delta_0(i,tmax) = Cal_delta_0(p, tmax, C, P);
        delta_1(i,tmax) = Cal_delta_1(p, tmax, C, P);
        
        % The values for each condition
        con1(i,tmax) = delta_1(i,tmax) - (1 + (1 - p) * sigma(i,tmax)) / 2;
        con2(i,tmax) = delta_1(i,tmax) - delta_0(i,tmax);
    end
end

%% Save the results
save('eq1.mat','con1','sigma','delta_0','delta_1');
save('eq2.mat','con2','sigma','delta_0','delta_1');

%% function - Cal_sigma
% Calculate sigma (see paper for definition)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   P - Probability of changing value
%   tmax - maximum transmission time
% Output:
%   sigma - sigma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sigma = Cal_sigma(p, tmax)
    sigma = (1 - (1 - p)^tmax) / (p * (1 - p * (1 - p)^(tmax - 1)));
end

%% function - Cal_delta_0
% Calculate delta_0 (see paper for definition)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   P - Probability of changing value
%   tmax - maximum transmission time
%   C - Instant costs
%   P - State transition probabilities
% Output:
%   delta_0 - delta_0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function delta_0 = Cal_delta_0(p, tmax, C, P)
    pi = zeros(1, tmax + 2);
    pi(1) = P(2,1) / tmax;
    for s = 1 : tmax - 1
        pi(s + 1) = (P(2,s+1) + (P(1,s+1) - P(2,s+1)) * P(2,1)) / tmax;
    end
    pi(tmax + 1) = (P(1,end) * P(2,1)) / tmax;
    pi(end) = P(2,end) * (1 - P(2,1)) / tmax;

    delta_0 = C(1) * pi(1);
    temp = 0;
    for s = 1 : tmax
        temp = temp + C(s+1) * pi(s+1);
    end
    temp = temp + tmax * (1 - (1 - p)^tmax) / p * pi(end);
    delta_0 = delta_0 + temp / (1 - P(2,end));
end

%% function - Cal_delta_1
% Calculate delta_1 (see paper for definition)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   P - Probability of changing value
%   tmax - maximum transmission time
%   C - Instant costs
%   P - State transition probabilities
% Output:
%   delta_1 - delta_1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function delta_1 = Cal_delta_1(p, tmax, C, P)
    pi = zeros(1, tmax + 2);
    pi(2) = ((P(2,1) + P(2,2)) * p) / (P(2,1) + p * tmax);
    for s = 2 : tmax - 1
        pi(s + 1) = (p * P(2,s+1)) / (P(2,1) + p * tmax);
    end
    pi(end) = (p * P(2,end)) / (P(2,1) + p * tmax);

    temp = 0;
    for s = 1 : tmax
        temp = temp + C(s+1) * pi(s+1);
    end
    temp = temp / (1 - P(2,end));
    delta_1 = temp + (tmax - tmax * (1 - p)^tmax) / (p - p * P(2,end)) * pi(end);
end

%% function - InstantCost
% Calculate the instant cost
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   s_init - AoII starting value
%   tmax - Maximum transmission time
%   p - Source process dynamics
%   P_chain - Multi-step state transition probability
% Output:
%   C - Instant costs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = InstantCost(s_init,tmax,p,P_chain)
    C = 0;
    % s = 0
    if s_init == 0
        for i = 1:tmax-1
            tmp = 0;
            for k = 1:i
               Pi = P_chain^(i-k);
               tmp = tmp + k*p*(1-p)^(k-1)*Pi(1,1);
            end
            C = C + tmp;
        end
    else
    % s > 0 
        for i = 1:tmax-1
            tmp = 0;
            for k = 1:i-1
               Pi = P_chain^(i-k);
               tmp = tmp + k*p*(1-p)^(k-1)*Pi(1,2);
            end
            C = C + tmp + (i+s_init)*(1-p)^i;
        end
        C = C + s_init;
    end
end
