%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file implements the relative value iteration algorithm to obtain the
% optimal policy for the considered problem.
% Author: Yutao Chen
% Updated: 07/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% System Parameters
p = 0.25;    % State Transition Probability in Markov Chain
tmax = 2;   % Maximum transmission time
smax = 300; % Finite the state space

epsilon = 0.01;   % Precision of RVIA
iter_max = 10000; % Maximum iteration of RVIA

%% Transmission time distribution
pb = 0.7;   % (Bernoulli) successful probability
pt = zeros(1,tmax+1);
for i = 1:tmax
    pt(i) = (1-pb)^(i-1) * pb;
end
pt(tmax+1) = 1 - sum(pt(1:tmax));

%% Derived variables
% Probabilities in system dynamic - Pr
P = zeros(1,tmax);
for i = 1:tmax
    P(i) = sum(pt(1:i));
end
Pr = zeros(1,tmax);
Pr(1) = 1-P(1);
for i = 2:tmax
    Pr(i) = (1-P(i)) / (1-P(i-1));
end

%% Relative Value Iteration Algorithm
fprintf('...RVI starts\n');

[delta_V,estimated_RVs] = RVI(smax, tmax, p, Pr, iter_max, epsilon);
thre_from_RVI = find(delta_V(:,1,1) < 0, 1);

fprintf('The threshold (from RVI) is %d\n', thre_from_RVI - 1);
fprintf('\n');

%% Function - RVI
% Apply the relative value function algorithm to obtain the optimal policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   smax - Upper bound on AoII
%   tmax - Update transmission time
%   p - Source process dynamics
%   Pr: Probabilities in systwm dynamic
%   iter_max - Maximum number of iterations
%   epsilon - Tolerance
% Output:
%   delta_V - The difference between the value function resulting from
%             taking two feasible actions
%   RV_Archive - Relative value function history
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [delta_V,RV_Archive] = RVI(smax, tmax, p, Pr, iter_max, epsilon)
    RV_old = zeros(smax+1, tmax, 3); % Value functions
    Trans = zeros(smax+1, tmax, 3);
    NoTrans = zeros(smax+1, tmax, 3);
    cnt = 0; % Counter
    RV_Archive = zeros(smax+1,tmax,3,100);
    
    % Main Loop
    while cnt <= iter_max
        % Feasible action (0 and 1)
        [Trans(:,1,1), NoTrans(:,1,1)] = TransTwoAction(p,Pr,RV_old,smax);
        % Feasible action (0)
        for t = 1:tmax-1
           [Trans(:,t+1,2), NoTrans(:,t+1,2)] = TransOneAction(p,Pr, ...
               RV_old,2,t,smax,tmax);
           [Trans(:,t+1,3), NoTrans(:,t+1,3)]= TransOneAction(p,Pr, ...
               RV_old,3,t,smax,tmax);
        end
        
        RV_new = min(NoTrans, Trans);
        RV_new = RV_new - RV_new(1,1,1);       % Relative
    
        RV_Archive(:,:,:,cnt+1) = RV_new;
    
        % Generate the threshold vector
        if cnt > 0
            if max(abs(RV_old - RV_new)) <= epsilon
                delta_V = Trans - NoTrans;
                break
            end
        end
        % One step forward
        RV_old = RV_new;
        cnt = cnt + 1;
    end
    fprintf('# of iteration ran = %d\n', cnt);
end

%% Function - TransOneAction
% State transitions when there is an update being transmitting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   p - Source process dynamics
%   Pr: Probabilities in systwm dynamic
%   RV_old - Current estimated value iteration
%   i - Indicator for the correctness transitting update (2-correct, 
%       3-incorrect)
%   t - Remaining transmission time of the current update
%   smax - Upper bound on AoII (truncation)
%   tmax: Mximum transmission time
% Output:
%   trans - Resulting value function when transmission happens
%   notrans - Resulting value function when no transmission happens
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [trans, notrans] = TransOneAction(p, Pr, RV_old, i, t, smax, tmax)
    notrans = zeros(smax+1,1);
    
    % s = 0 and i = 0
    if i == 2
        if t == tmax-1
            notrans(1) = (1-p) * RV_old(1,1,1) + p * RV_old(2,1,1);
        else
            notrans(1) = Pr(t+1)*(1-p) * RV_old(1,t+2,i) + ...
                         Pr(t+1)*p * RV_old(2,t+2,i) + ...
                         (1-Pr(t+1))*(1-p) * RV_old(1,1,1) + ...
                         (1-Pr(t+1))*p * RV_old(2,1,1);
        end
    end
    % s = 0 and i = 1
    if i == 3
        if t == tmax-1
            notrans(1) = (1-Pr(t+1))* p * RV_old(1,1,1) + ...
                         (1-Pr(t+1))*(1-p) * RV_old(2,1,1) + ... 
                         Pr(t+1) * p * RV_old(2,1,1) + ...
                         Pr(t+1) * (1-p) * RV_old(1,1,1);
        else
            notrans(1) = Pr(t+1)*(1-p) * RV_old(1,t+2,i) + ...
                         Pr(t+1)*p * RV_old(2,t+2,i) + ...
                         (1-Pr(t+1))* p * RV_old(1,1,1) + ...
                         (1-Pr(t+1))*(1-p) * RV_old(2,1,1);
        end
    end

    % s > 0 and i = 0
    if i == 2
        for s = 2:smax
            if t == tmax-1
                notrans(s) = s-1 + (1-p)*RV_old(s+1,1,1) + p*RV_old(1,1,1);
            else
                notrans(s) = s - 1 + Pr(t+1)*(1-p) * ...
                             RV_old(s+1,t+2,i) + Pr(t+1)*p * ...
                             RV_old(1,t+2,i) + (1-Pr(t+1))*(1-p) * ...
                             RV_old(s+1,1,1) + (1-Pr(t+1))*p * ...
                             RV_old(1,1,1);
            end
        end
        if t == tmax-1
            notrans(smax+1) = smax + (1-p)*RV_old(smax+1,1,1) + ...
                                     p*RV_old(1,1,1);
        else
            notrans(smax+1) = smax + Pr(t+1)*(1-p) * ...
                              RV_old(smax+1,t+2,i) + Pr(t+1)*p * ...
                              RV_old(1,t+2,i) + (1-Pr(t+1))*(1-p) * ...
                              RV_old(smax+1,1,1) + (1-Pr(t+1))*p * ...
                              RV_old(1,1,1);
        end
    end
    % s > 0 and i = 1
    if i == 3
        for s = 2:smax
            if t == tmax-1
                notrans(s) = s-1 + (1-Pr(t+1))*p * RV_old(s+1,1,1) + ...
                                   (1-Pr(t+1))*(1-p) * RV_old(1,1,1) + ...
                                   Pr(t+1)*p*RV_old(1,1,1) + ...
                                   Pr(t+1)*(1-p)*RV_old(s+1,1,1);
            else
                notrans(s) = s-1 + Pr(t+1)*(1-p) * RV_old(s+1,t+2,i) + ...
                                   Pr(t+1)*p * RV_old(1,t+2,i) + ...
                                   (1-Pr(t+1))* p * RV_old(s+1,1,1) + ...
                                   (1-Pr(t+1))* (1-p) * RV_old(1,1,1);
            end
        end
        if t == tmax-1
            notrans(smax+1) = smax + (1-Pr(t+1))*p* RV_old(smax+1,1,1) + ...
                                     (1-Pr(t+1))*(1-p)* RV_old(1,1,1) + ...
                                     Pr(t+1)*p*RV_old(1,1,1) + ...
                                     Pr(t+1)*(1-p)*RV_old(smax+1,1,1);
        else
            notrans(smax+1) = smax + Pr(t+1)*(1-p) * RV_old(smax+1,t+2,i) +...
                                     Pr(t+1)*p * RV_old(1,t+2,i) + ...
                                     (1-Pr(t+1))* p * RV_old(smax+1,1,1) + ...
                                     (1-Pr(t+1))* (1-p) * RV_old(1,1,1);
        end
    end

    trans = notrans;  % By our definition, we let them to be the same
end

%% Function - TransTwoAction
% State transitions when there is no update being transmitting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   p - Source process dynamics
%   Pr - Probabilities in systwm dynamic
%   RV_old - Current estimated value iteration
%   smax - Upper bound on AoII (truncation)
% Output:
%   trans - Resulting value function when transmission happens
%   notrans - Resulting value function when no transmission happens
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [trans,notrans] = TransTwoAction(p, Pr, RV_old, smax)
    trans = zeros(smax+1,1);
    notrans = zeros(smax+1,1);
    
    % s = 0
    trans(1) = Pr(1)*(1-p) * RV_old(1,2,2) + Pr(1)*p * RV_old(2,2,2) + ...
               (1-Pr(1))*(1-p) * RV_old(1,1,1) + ...
               (1-Pr(1))*p * RV_old(2,1,1);
    notrans(1) = (1-p) * RV_old(1,1,1) + p * RV_old(2,1,1);
    
    % s > 0
    for s = 2:smax
        trans(s) = s - 1 + Pr(1)*(1-p) * RV_old(s+1,2,3) + ...
                   Pr(1)*p * RV_old(1,2,3) + ...
                   (1-Pr(1)) * p * RV_old(s+1,1,1) + ...
                   (1-Pr(1))*(1-p) * RV_old(1,1,1);
        notrans(s) = s - 1 + (1-p) * RV_old(s+1,1,1) + p * RV_old(1,1,1);
    end
    trans(smax+1) = smax + Pr(1)*(1-p) * RV_old(smax+1,2,3) + ...
                    Pr(1)*p * RV_old(1,2,3) + (1-Pr(1)) * p * ...
                    RV_old(smax+1,1,1) + (1-Pr(1))*(1-p) * RV_old(1,1,1);
    notrans(smax+1) = smax + (1-p) * RV_old(smax+1,1,1) + ...
                      p * RV_old(1,1,1); 
end
