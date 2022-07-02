%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file verifies Condition 1 in the paper.
% Author: Yutao Chen
% Updated: 07/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% Parameters
smax = 300;  % Truncate the state space

%% Store Results
tmax_range = 15;
value_intervals = 21;
sigma = zeros(value_intervals,tmax_range);
AoII0 = zeros(value_intervals,tmax_range);
AoII1 = zeros(value_intervals,tmax_range);
eq1 = zeros(value_intervals,tmax_range);
eq2 = zeros(value_intervals,tmax_range);

%% Main Loop
for tmax = 2 : tmax_range
    for interval = 1 : value_intervals
        p = (interval - 1) * 0.05 / 2;
%         p = 0.35;
        pb = 0.6;
%         pb = (interval - 1) * 0.05;

        P_chain = [(1-p),p;p,(1-p)];

        pt = zeros(1,tmax+1);
        for i = 1:tmax
            pt(i) = (1-pb)^(i-1) * pb;
        end
        pt(tmax+1) = 1 - sum(pt(1:tmax));

        % Expected transmission time - ET
        ET = sum(pt(1:tmax) .* (1:tmax)) + tmax * pt(tmax+1);
        
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

        % Compact Cost - C_rand
        C_det = zeros(smax+1,tmax);
        for i = 0:smax
            for t = 1:tmax
                C_det(i+1,t) = InstantCost(i,t,p,P_chain);
            end
        end
        C_rand = zeros(smax+1,1);
        for s = 0:smax
            C_rand(s+1) = sum(pt(1:tmax) .* C_det(s+1,:)) + ...
                          pt(tmax+1) * C_det(s+1,end);
        end

        % Compact state transition probabilities - P_rand
        P = zeros(101,101,tmax); % delta x delta' x t
        for t = 1:tmax
            tmp = P_chain^t;
            P(1,1,t) = tmp(1,1);
            for k = 1:t
               tmp = P_chain^(t-k);
               P(1,k+1,t) = tmp(1,1) * p * (1-p)^(k-1);
            end
            for s = 1 : 101
                tmp = P_chain^t;
                P(s+1,1,t) = tmp(1,1);
                tmp = P_chain^(t-1);
                P(s+1,2,t) = tmp(2,1) * (1-p);
                for k = 2:t-1
                   tmp = P_chain^(t-k);
                   P(s+1,k+1,t) = tmp(2,1) * p^2 * (1-p)^(k-2);
                end
                P(s+1,s+t+1,t) = p * (1-p)^(t-1);
            end
        end

        P_plus = zeros(101,101);
        tmp = P_chain^tmax;
        P_plus(1,1) = tmp(1,1);
        for k = 1:tmax
           tmp = P_chain^(tmax-k);
           P_plus(1,k+1) = tmp(1,1) * p * (1-p)^(k-1);
        end
        for s = 1 : 101
            tmp = P_chain^tmax;
            P_plus(s+1,1) = tmp(2,1);
            for k = 1:tmax-1
               tmp = P_chain^(tmax-k);
               P_plus(s+1,k+1) = tmp(2,1) * p * (1-p)^(k-1);
            end
            P_plus(s+1,s+tmax+1) = (1-p)^tmax;
        end

        P_rand = zeros(101,101+tmax); % delta x delta'
        for delta = 0 : 100
            for delta_prime = 0 : 100 + tmax
                temp = 0;
                for t = 1:tmax
                    temp = temp + pt(t) * P(delta+1,delta_prime+1,t);
                end
                P_rand(delta+1,delta_prime+1) = temp + pt(tmax+1) * ...
                                             P_plus(delta+1,delta_prime+1);
            end
        end

        % Upsilon
        U = zeros(101,tmax);
        for t = 1 : tmax
            for delta = t:100
                U(delta+1,t) = pt(t) * P(delta-t+1,delta+1,t) + ...
                               pt(tmax+1) * P_plus(delta-t+1,delta+1);
            end
        end


        % Results
        sigma(interval,tmax-1) = Cal_sigma(p, pt, tmax);
        
        [AoII0(interval,tmax-1),~,~] = PerformanceTheo(p, P_rand, ...
                                           tmax, 0, ET, 1, C_rand, pt, U);

        [AoII1(interval,tmax-1),~,~] = PerformanceTheo(p, P_rand, ...
                                           tmax, 1, ET, 2, C_rand, pt, U);

        eq1(interval,tmax-1) = AoII1(interval,tmax-1) - ...
                               AoII0(interval,tmax-1);
        eq2(interval,tmax-1) = AoII1(interval,tmax-1) - ...
                               (1+(1-p)*sigma(interval,tmax-1))/2;
    end
end

%% Save the data
p_pool = linspace(0,0.5,value_intervals);
pb_pool = linspace(0,1,value_intervals);
save('pb06.mat','eq1','eq2','p_pool');
% save('p035.mat','eq1','eq2','pb_pool');

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
    numerator = 0;
    for i = 1 : tmax
        numerator = numerator + pt(i) * ((1-(1-p)^i)/p);
    end
    numerator = numerator + pt(tmax+1) * ((1-(1-p)^tmax)/p);
    denominator = 0;
    for t = 1 : tmax
        denominator = denominator + pt(t) * p * (1-p)^(t-1);
    end
    denominator = denominator + pt(tmax+1) * (1-p)^tmax;
    sigma = numerator / (1 - denominator);
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

%% Function - Theoretical
% Calculate the theoretical value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   p - Source process dynamics
%   P_rand: Compact stste transition probabilities
%   tmax - update transmission time
%   tau - Threshold
%   ET - Expected transmission time
%   policy_num - Policy index
%   C_rand - Compact state transition probabilities
%   pt - Transmission time distribution
%   P - State transition probabilities
% Output:
%   AoII - Expected Age of Incorrect Information
%   dist - The stationary distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [AoII,dist,bigpi] = PerformanceTheo(p, P_rand, tmax, tau, ET, ...
                                             policy_num, C_rand, pt, U)
    % Lazy policy
    if policy_num == 3
        AoII = 1 / (2 * p);
        dist = 0;
    
    % Zero Wait policy
    elseif policy_num == 1
        dist = zeros(1,tmax+2);  % 0 - tmax + Pi
        dist(1) = P_rand(2,1) / (ET * (1-P_rand(1,1) + P_rand(2,1)));
        for delta = 1 : tmax - 1
            temp1 = 0;
            temp2 = 0;
            for s = 0 : delta - 1
                temp1 = temp1 + dist(s+1) * P_rand(s+1,delta+1);
                temp2 = temp2 + dist(s+1);
            end
            dist(delta+1) = temp1 + P_rand(delta+1,delta+1) * ...
                            (1/ET - temp2);
        end
        
        temp = 0;
        for i = 0:tmax-1
            temp = temp + P_rand(i+1,tmax+1) * dist(i+1);
        end
        dist(tmax+1) = temp;
    
        numerator = 0;
        for i = 0:tmax-1
            temp = 0;
            for k = 0:i
                temp = temp + P_rand(i+1,tmax+k+1);
            end
            numerator = numerator + temp * dist(i+1);
        end
        temp = 0;
        for i = 1:tmax
            temp = temp + P_rand(tmax+1,tmax+i+1);
        end
        denominator = 1 - temp;
        dist(tmax+2) = numerator / denominator;

        % Expected AoII
        AoII = 0;
        for s = 0:tmax
            AoII = AoII + C_rand(s+1) * dist(s+1);
        end

        bigpi = zeros(tmax);
        for t = 1:tmax
            temp = 0;
            for s = tmax + 1 - t : tmax - 1
                temp = temp + U(s+t+1,t) * dist(s+1);
            end
            bigpi(t) = temp + U(tmax+t+1,t)* dist(tmax+2);
        end
        delta_prime = zeros(tmax);
        for t = 1 : tmax
            temp = 0;
            for i = 1 : tmax
                temp = temp + (pt(i)*(t-t*(1-p)^i))/p;
            end
            delta_prime(t) = temp + (pt(tmax+1)*(t-t*(1-p)^tmax))/p;
        end
        numerator = 0;
        for t = 1:tmax
            temp = 0;
            for s = tmax + 1 - t : tmax
                temp = temp + U(s+t+1,t) * C_rand(s+1) * dist(s+1);
            end
            numerator = numerator + temp + bigpi(t) * delta_prime(t);
        end
        denominator = 0;
        for t = 1 : tmax
            denominator = denominator + U(tmax+t+2,t);
        end
        denominator = 1 - denominator;
        AoII = AoII + numerator / denominator;
    
    % Threshold policy
    elseif policy_num == 2
        if tau == 1
            dist = zeros(1,tmax+2);
            dist(1) =  P_rand(2,1) / (p * ET + P_rand(2,1));
            
            dist(2) = (p * (P_rand(2,1) + P_rand(2,2))) / ...
                      (p * ET + P_rand(2,1));
    
            for s = 2:tmax-1
                temp1 = 0;
                temp2 = 0;
                for i = 1:s-1
                    temp1 = temp1 + P_rand(i+1,s+1) * dist(i+1);
                    temp2 = temp2 + dist(i+1);
                end
                dist(s+1) = temp1 + P_rand(s+1,s+1) * ...
                            (((1-dist(1))/ET)-temp2);
            end
        
            for s = 1:tmax-1
                dist(tmax+1) = dist(tmax+1) + P_rand(s+1,tmax+1) * ...
                               dist(s+1);
            end
        
            numerator = 0;
            for s = 1:tmax
                temp = 0;
                for k = 1:s
                    temp = temp + P_rand(s+1,tmax+k+1);
                end
                numerator = numerator + temp * dist(s+1);
            end
            temp = 0;
            for s = 1:tmax
                temp = temp + P_rand(tmax+2,tmax+s+2);
            end
            denominator = 1 - temp;
            dist(end) = numerator / denominator;

            % Expected AoII
            omega = tmax + 1;
            AoII = 0;
            for s = 1:omega-1
                AoII = AoII + C_rand(s+1) * dist(s+1);
            end
            bigpi = zeros(tmax);
            for t = 1:tmax
                temp = 0;
                for s = omega - t : omega - 1
                    temp = temp + U(s+t+1,t) * dist(s+1);
                end
                bigpi(t) = temp + U(omega+t+1,t)* dist(end);
            end
            delta_prime = zeros(tmax);
            for t = 1 : tmax
                temp = 0;
                for i = 1 : tmax
                    temp = temp + (pt(i)*(t-t*(1-p)^i))/p;
                end
                delta_prime(t) = temp + (pt(tmax+1)*(t-t*(1-p)^tmax))/p;
            end
            numerator = 0;
            for t = 1:tmax
                temp = 0;
                for s = omega - t : omega - 1
                    temp = temp + U(s+t+1,t) * C_rand(s+1) * dist(s+1);
                end
                numerator = numerator + temp + bigpi(t) * delta_prime(t);
            end
            denominator = 0;
            for t = 1 : tmax
                denominator = denominator + U(omega+t+1,t);
            end
            denominator = 1 - denominator;
            AoII = AoII + numerator / denominator;
    
        else % tau >= 2
            omega = tau + tmax;
        
            % Stationary Distribution
            Pi = zeros(omega + 2, omega + 1);
            
            % pi0
            row = 1;
            Pi(row,1) = Pi(row,1) + 1 - p;
            for i = 1:tau-1
                Pi(row,i+1) = Pi(row,i+1) + p;
            end 
            for i = tau:omega
                Pi(row,i+1) = Pi(row,i+1)+ P_rand(tau+1,1);
            end
            
            % pi1
            row = 2;
            Pi(row,1) = Pi(row,1) + p;
            for i = tau:omega
                Pi(row,i+1) = Pi(row,i+1) + P_rand(tau+1,2);
            end
            
            % 2 <= delta <= tmax-1
            for delta = 2:tmax-1
               row = delta + 1;
               if delta-1 < tau
                   Pi(row,delta) = Pi(row,delta) + 1 - p;
                   for i = tau:omega
                        Pi(row,i+1) = Pi(row,i+1) + P_rand(tau+1,delta+1);
                   end
               end
               if delta-1 >= tau
                    for i = tau:delta-1
                        Pi(row,i+1) = Pi(row,i+1) + P_rand(i+1,delta+1);
                    end
                    for i = delta:omega
                       Pi(row,i+1) = Pi(row,i+1) + P_rand(delta+1,delta+1);
                    end
               end
            end
            
            % tmax <= delta <= omega - 1
            for delta = tmax : omega - 1
                row = delta + 1;
                if delta - 1 < tau
                   Pi(row,delta) = Pi(row,delta) + 1 - p; 
                else
                    for i = tau : delta - 1
                        Pi(row,i+1) = Pi(row,i+1) + P_rand(i+1,delta+1);
                    end
                end
            end
            
            % PI (delta = omega)
            row = omega + 1;
            for i = tau : omega - 1
                temp = 0;
                for k = tau:i
                    temp = temp + P_rand(i+1,tmax+k+1);
                end
                Pi(row,i+1) = Pi(row,i+1) + temp;
            end
            temp = 0;
            for i = 1:tmax
                temp = temp + P_rand(omega+1,omega+i+1);
            end
            Pi(row,omega+1) = Pi(row,omega+1) + temp;
            
            % Sum to one
            together = ones(1,omega+1);
            together(tau+1:omega+1) = ET;
            Pi(end,:) = together;
            
            for i = 0:omega
                Pi(i+1,i+1) =  Pi(i+1,i+1) - 1;
            end
            
            B = zeros(omega+1,1);
            B(end) = 1;
            
            dist = transpose(linsolve(Pi(2:end,:),B));

            % Expected AoII
            AoII = 0;
            for s = 0 : tau - 1
                AoII = AoII + s * dist(s+1);
            end
            for s = tau : omega-1
                AoII = AoII + C_rand(s+1) * dist(s+1);
            end
            bigpi = zeros(tmax);
            for t = 1:tmax
                temp = 0;
                for s = omega - t : omega - 1
                    temp = temp + U(s+t+1,t) * dist(s+1);
                end
                bigpi(t) = temp + U(omega+t+1,t)* dist(end);
            end
            delta_prime = zeros(tmax);
            for t = 1 : tmax
                temp = 0;
                for i = 1 : tmax
                    temp = temp + (pt(i)*(t-t*(1-p)^i))/p;
                end
                delta_prime(t) = temp + (pt(tmax+1)*(t-t*(1-p)^tmax))/p;
            end
            numerator = 0;
            for t = 1:tmax
                temp = 0;
                for s = omega - t : omega - 1
                    temp = temp + U(s+t+1,t) * C_rand(s+1) * dist(s+1);
                end
                numerator = numerator + temp + bigpi(t) * delta_prime(t);
            end
            denominator = 0;
            for t = 1 : tmax
                denominator = denominator + U(omega+t+1,t);
            end
            denominator = 1 - denominator;
            AoII = AoII + numerator / denominator;
        end
    else
        fprintf('Invalid policy index.');
    end
end
