%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file verifies Condition 1 in the paper when the transmission time
% for an update is random under various system parameters.
% See Section V.A of the paper
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% Parameters
smax = 300;  % Truncate the state space

%% Store Results
tmax_range = 15;
value_intervals = 21;
sigma = zeros(value_intervals,tmax_range);
delta_0 = zeros(value_intervals,tmax_range);
delta_1 = zeros(value_intervals,tmax_range);
con1 = zeros(value_intervals,tmax_range);
con2 = zeros(value_intervals,tmax_range);

%% Main Loop
for tmax = 1 : 15
    for i = 1 : 21
        % System parameters
%         p = (i - 1) * 0.05 / 2;
        p = 0.35;
%         pb = 0.6;
        pb = (i - 1) * 0.05;
        P_chain = [(1-p),p;p,(1-p)];
        
        % CDF of transmission time
        pt = zeros(1,tmax);
        for j = 1:tmax-1
            pt(j) = (1-pb)^(j-1) * pb;
        end
        pt(tmax) = (1-pb)^(tmax-1);

        % Expected transmission time - ET
        ET = sum(pt .* (1:tmax));
        
        % Probabilities in systwm dynamic - Pr
        P = zeros(1,tmax);
        for j = 1:tmax
            P(j) = sum(pt(1:j));
        end
        Pr = zeros(1,tmax);
        Pr(1) = 1-P(1);
        for j = 2:tmax
            Pr(j) = (1-P(j)) / (1-P(j-1));
        end
        Pr(tmax) = 0; % To eliminate possible round-down errors
        
        % Compact Cost - C_rand
        C_det = zeros(smax+1,tmax);
        for j = 0:smax
            for t = 1:tmax
                C_det(j+1,t) = InstantCost(j,t,p,P_chain);
            end
        end
        C_rand = zeros(smax+1,1);
        for s = 0:smax
            C_rand(s+1) = sum(pt .* C_det(s+1,:));
        end
        
        % Compact state transition probabilities - P_rand
        P = zeros(2,tmax+1,tmax);
        for t = 1:tmax
            tmp = P_chain^t;
            P(1,1,t) = tmp(1,1);
            P(2,1,t) = tmp(1,1);
            for s = 2:t+1
               tmp = P_chain^(t-s+1);
               P(1,s,t) = tmp(1,1) * p * (1-p)^(s-2);
            end
            tmp = P_chain^(t-1);
            P(2,2,t) = tmp(1,2) * (1-p);
            for s = 3:t
               tmp = P_chain^(t-s+1);
               P(2,s,t) = tmp(1,2) * p^2 * (1-p)^(s-3);
            end
            P(2,t+1,t) = p * (1-p)^(t-1);
        end
        P_rand = zeros(100,100);
        for t = 1:tmax
            for j = 0:t
                P_rand(1,j+1) = P_rand(1,j+1) + pt(t) * P(1,j+1,t);
            end
        end
        for s = 1:99
            for t = 1:tmax
                for j = 0:t-1
                    P_rand(s+1,j+1) = P_rand(s+1,j+1) + pt(t) * P(2,j+1,t);
                end
                if t+s+1 <= 100
                    P_rand(s+1,t+s+1) = P_rand(s+1,t+s+1) + pt(t) * P(2,t+1,t);
                end
            end
        end
        
        % Auxiliary quantities
        sigma(i,tmax) = Cal_sigma(p, pt, tmax);
        delta_0(i,tmax) = Cal_delta_0(p, tmax, C_rand, P, P_rand, ET, pt);
        delta_1(i,tmax) = Cal_delta_1(p, tmax, C_rand, P, P_rand, ET, pt);
        
        % The values for each condition
        con1(i,tmax) = delta_1(i,tmax) - (1 + (1 - p) * sigma(i,tmax)) / 2;
        con2(i,tmax) = delta_1(i,tmax) - delta_0(i,tmax);
    end
end

%% Save the results
% save('eq1_p.mat','con1');
% save('eq2_p.mat','con2');

save('eq1_pb.mat','con1');
save('eq2_pb.mat','con2');

%% function - Cal_sigma
% Calculate sigma (see paper for definition)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   p - Probability of changing value
%   pt - transmission time CDF
%   tmax - maximum transmission time
% Output:
%   sigma - sigma
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sigma = Cal_sigma(p, pt, tmax)
    temp1 = 0;
    temp2 = 0;
    for t = 1:tmax
        temp1 = temp1 + pt(t) * (1-(1-p)^t) / p;
        temp2 = temp2 + pt(t) * p * (1-p)^(t-1);
    end
    sigma = temp1 / (1-temp2);
end

%% function - Cal_delta_0
% Calculate delta_0 (see paper for definition)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   p - Probability of changing value
%   tmax - Maximum transmission time
%   C_rand - Compact Cost
%   P - Probabilities in systwm dynamic
%   P_rand - Compact state transition probabilities
%   ET - Expected transmission time
%   pt - Transmission time CDF
% Output:
%   delta_0 - delta_0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function delta_0 = Cal_delta_0(p, tmax, C_rand, P, P_rand, ET, pt)
    pi = zeros(1,tmax+2);
    pi(1) =  P_rand(1,1) / ET;

    for s = 1:tmax-1
        temp1 = 0;
        temp2 = 0;
        for i = 0:s-1
            temp1 = temp1 + P_rand(i+1,s+1) * pi(i+1);
            temp2 = temp2 + pi(i+1);
        end
        pi(s+1) = temp1 + P_rand(s+1,s+1)*((1/ET)-temp2);
    end

    for s = 0:tmax-1
        pi(tmax+1) = pi(tmax+1) + P_rand(s+1,tmax+1) * pi(s+1);
    end

    numerator = 0;
    for s = 1:tmax
        temp = 0;
        for k = 1:s
            temp = temp + P_rand(s+1,tmax+k+1);
        end
        numerator = numerator + temp * pi(s+1);
    end
    temp = 0;
    for s = 1:tmax
        temp = temp + P_rand(tmax+1,tmax+s+1);
    end
    denominator = 1 - temp;
    pi(end) = numerator / denominator;

    % Expected AoII
    delta_0 = 0;
    for s = 0:tmax
        delta_0 = delta_0 + C_rand(s+1) * pi(s+1);
    end
    temp1 = 0;
    temp2 = 0;
    for t = 1:tmax
        temp3 = 0;
        for s = tmax+1-t:tmax
            temp3 = temp3 + C_rand(s+1) * pi(s+1);
        end
        temp1 = temp1 + pt(t) * P(2,t+1,t) * temp3;
        temp2 = temp2 + pt(t) * P(2,t+1,t);
    end
    s_prime = zeros(1,tmax);
    bigpi = zeros(1,tmax);
    for t = 1:tmax
        temp = 0;
        for s = 1:tmax
            temp = temp + pt(s) * (t-t*(1-p)^s) / p;
        end
        s_prime(t) = temp;
        temp = 0;
        for s = tmax+1-t:tmax
            temp = temp + pi(s+1);
        end
        temp = temp + pi(end);
        bigpi(t) = pt(t) * P(2,t+1,t) * temp;
    end
    delta_0 = delta_0 + (temp1 + sum(s_prime .* bigpi)) / (1 - temp2);
end

%% function - Cal_delta_1
% Calculate delta_1 (see paper for definition)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   p - Probability of changing value
%   tmax - Maximum transmission time
%   C_rand - Compact Cost
%   P - Probabilities in systwm dynamic
%   P_rand - Compact state transition probabilities
%   ET - Expected transmission time
%   pt - Transmission time CDF
% Output:
%   delta_1 - delta_1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function delta_1 = Cal_delta_1(p, tmax, C_rand, P, P_rand, ET, pt)
    pi = zeros(1,tmax+2);
    pi(1) =  P_rand(2,1) / (p * ET + P_rand(2,1));
    
    pi(2) = (p * (P_rand(2,1) + P_rand(2,2))) / (p * ET + P_rand(2,1));

    for s = 2:tmax-1
        temp1 = 0;
        temp2 = 0;
        for i = 1:s-1
            temp1 = temp1 + P_rand(i+1,s+1) * pi(i+1);
            temp2 = temp2 + pi(i+1);
        end
        pi(s+1) = temp1 + P_rand(s+1,s+1)*(((1-pi(1))/ET)-temp2);
    end

    for s = 1:tmax-1
        pi(tmax+1) = pi(tmax+1) + P_rand(s+1,tmax+1) * pi(s+1);
    end

    numerator = 0;
    for s = 1:tmax
        temp = 0;
        for k = 1:s
            temp = temp + P_rand(s+1,tmax+k+1);
        end
        numerator = numerator + temp * pi(s+1);
    end
    temp = 0;
    for s = 1:tmax
        temp = temp + P_rand(tmax+1,tmax+s+1);
    end
    denominator = 1 - temp;
    pi(end) = numerator / denominator;

    % Expected AoII
    delta_1 = 0;
    for s = 1:tmax
        delta_1 = delta_1 + C_rand(s+1) * pi(s+1);
    end
    temp1 = 0;
    temp2 = 0;
    for t = 1:tmax
        temp3 = 0;
        for s = tmax+1-t:tmax
            temp3 = temp3 + C_rand(s+1) * pi(s+1);
        end
        temp1 = temp1 + pt(t) * P(2,t+1,t) * temp3;
        temp2 = temp2 + pt(t) * P(2,t+1,t);
    end
    s_prime = zeros(1,tmax);
    bigpi = zeros(1,tmax);
    for t = 1:tmax
        temp = 0;
        for s = 1:tmax
            temp = temp + pt(s) * (t-t*(1-p)^s) / p;
        end
        s_prime(t) = temp;
        temp = 0;
        for s = tmax+1-t:tmax
            temp = temp + pi(s+1);
        end
        temp = temp + pi(end);
        bigpi(t) = pt(t) * P(2,t+1,t) * temp;
    end
    delta_1 = delta_1 + (temp1 + sum(s_prime .* bigpi)) / (1 - temp2);
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
