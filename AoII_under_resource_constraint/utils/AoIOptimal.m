%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file evaluates the performance of the AoI-optimal policy.
% See Section IV of the paper
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% Parameter
N = 7;         % number of states
p_pool = 0.2;  % Probability of changing state
Ps = 0.8;      % Probability of successful transmission
alpha = 0.02;  % Power budget

AoII_pool = zeros(1,length(p_pool));  % Store the resulting AoII

%% Main Loop
for k = 1:length(p_pool)
    P = p_pool(k);
    p = 1 - Ps;
    
    % The AoI-optimal policy
    delta_alpha = (1/alpha - p) / (1-p);
    delta_1 = floor(delta_alpha);
    delta_2 = ceil(delta_alpha);
    if delta_1 == delta_2
        threshold_1 = delta_alpha;
        threshold_2 = delta_alpha;
        mu = 1;
    else
        threshold_1 = delta_1;
        threshold_2 = delta_2;
        rate_1 = 1 / (threshold_1*(1-p) + p);
        rate_2 = 1 / (threshold_2*(1-p) + p);
        mu = (alpha - rate_2) / (rate_1 - rate_2);
    end

    % Obtain resulting AoII by simulation
    run = 50;       % Number of runs
    epoch = 10000;  % Epochs in each run
    AoII = 0;       % Expected AoII
    Ratio = 0;      % Expected Transmission Ratio
    
    for i = 1:run
        penalty = 0;
        AoI = 0;
        diff = 0;
        AoII_in_run = 0;
        transmission_cntr = 0;
        for j = 1:epoch
            if AoI >= threshold_2
                action = 1;
            elseif AoI < threshold_1
                action = 0;
            else
                action = binornd(1,mu);
            end
            transmission_cntr  = transmission_cntr + action;
            if action == 0
                [diff,penalty,AoI] = trans(N,P,diff,penalty,AoI,0);
            else
                r = binornd(1,Ps);
                [diff,penalty,AoI] = trans(N,P,diff,penalty,AoI,r);
            end
            AoII_in_run = AoII_in_run + penalty; 
        end
        AoII_out_run = AoII_in_run / epoch;
        AoII = AoII + AoII_out_run;
        Ratio_out_run = transmission_cntr / epoch;
        Ratio = Ratio + Ratio_out_run;
    end
    AoII = AoII / run;
    Ratio = Ratio / run;
    fprintf('Expected AoII: %.4f\n', AoII);
    fprintf('Expected Transmission Ratio: %.4f\n', Ratio);
    AoII_pool(k) = AoII;
end

%% Save the results
filename = 'OptimalPolicies-AoI.mat';
save(filename, 'AoII_pool','N','p_pool','Ps','alpha');

%% Function - trans
% Return the system status for the given current state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   N - number of states
%   P - probability of changing value
%   old_diff - current difference between the source and the estimate
%   old_penalty - current penalty
%   old_aoi - current AoI
%   channel_realization - 1 if transmission succeeds and 0 otherwise
% Output:
%   new_diff - difference at next time slot
%   new_penalty - penalty at next time slot
%   new_aoi - AoI at next time slot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [new_diff, new_penalty,new_aoi] = trans( N,P,old_diff, ...
                                                  old_penalty,cur_aoi, ...
                                                  channel_realization)
    % When the transmission fails
    if channel_realization == 0
        if old_diff == 0
            new_diff = randsrc(1,1,[old_diff,old_diff+1;1-2*P,2*P]);
        elseif old_diff == N-1
            new_diff = randsrc(1,1,[old_diff,old_diff-1;1-2*P,2*P]);
        else
            new_diff = randsrc(1,1,[old_diff-1,old_diff,old_diff+1; ...
                                    P,1-2*P,P]);
        end
        if new_diff == 0
            new_penalty = 0;
        else
            new_penalty = old_penalty + new_diff;
        end
        new_aoi = cur_aoi + 1;
    % When the transmission succeeds
    else
        new_diff = randsrc(1,1,[0,1;1-2*P,2*P]);
        if new_diff == 0
            new_penalty = 0;
        else
            new_penalty = 1;
        end
        new_aoi = 1;
    end
end
