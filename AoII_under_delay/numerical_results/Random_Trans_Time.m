%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file generates the numerical results as presented in the paper in
% the case where the transmission time is random. See Section V.B.
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% Store the results
AoII = zeros(3,21,5);

%% Main Loop
for y = 1:5
    tmax = y * 2;
    for z = 1:21
        % System Parameters
%         p = (z - 1) * 0.05;
        p = 0.35;  % State Transition Probability in Markov Chain
%         pb = 0.6;
        pb = (z - 1) * 0.05;  % (Bernoulli) successful probability
        
        smax = 300; % Finite the state space
        P_chain = [(1-p),p;p,(1-p)];
        
        % Policy parameters
        policy_num = 2; % Policy index (1-ZeroWait; 2-Threshold; 3-Lazy)
        thre = 1;       % Threshold in Threshold policy (>0)
        tau = thre;     % Renaming due to combination of multiple files
        
        % Performance simulation parameters
        run = 15;      % Number of runs
        epoch = 25000; % Epochs in each run
        
        % Geometric distribution
        pt = zeros(1,tmax);
        for i = 1:tmax-1
            pt(i) = (1-pb)^(i-1) * pb;
        end
        pt(tmax) = (1-pb)^(tmax-1);

        % Expected transmission time - ET
        ET = sum(pt .* (1:tmax));
        
        % Probabilities in systwm dynamic - Pr
        P = zeros(1,tmax);
        for i = 1:tmax
            P(i) = sum(pt(1:i));
        end
        Pr = zeros(1,tmax);
        Pr(1) = 1-P(1);
        for i = 2:tmax
            Pr(i) = (1-P(i)) / (1-P(i-1));
        end
        Pr(tmax) = 0; % To eliminate possible round-down errors
        
        % Compact Cost - C_rand
        C_det = zeros(smax+1,tmax);
        for i = 0:smax
            for t = 1:tmax
                C_det(i+1,t) = InstantCost(i,t,p,P_chain);
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
                    P_rand(s+1,j+1) = P_rand(s+1,j+1) + pt(t) * ...
                                      P(2,j+1,t);
                end
                if t+s+1 <= 100
                    P_rand(s+1,t+s+1) = P_rand(s+1,t+s+1) + pt(t) * ...
                                        P(2,t+1,t);
                end
            end
        end
        
        % Simulation
        [sim_penalty, sim_pi] = PerformanceSim(epoch, run, thre, ...
                                               policy_num, p, tmax, ...
                                               Pr, tau);
        AoII(1,z,y) = sim_penalty;
        
        % Theoretical
        [theo_penalty,theo_pi] = PerformanceTheo(p, P_rand, tmax, tau, ...
                                                 ET, policy_num, ...
                                                 C_rand, pt, P);
        AoII(2,z,y) = theo_penalty;

        % Accuracy
        AoII(3,z,y) = 100 * abs(sim_penalty - theo_penalty) / sim_penalty;
    end
end  

%% Store the results
save('AoII_p035.mat', 'AoII');

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
function [theo_penalty,pi] = PerformanceTheo(p, P_rand, tmax, tau, ...
                                             ET, policy_num, C_rand, pt, P)
    % Lazy policy
    if policy_num == 3
        theo_penalty = 1 / (2 * p);
        pi = 0;
    
    % Zero Wait policy
    elseif policy_num == 1
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
        theo_penalty = 0;
        for s = 0:tmax
            theo_penalty = theo_penalty + C_rand(s+1) * pi(s+1);
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
        theo_penalty = theo_penalty + (temp1 + sum(s_prime .* bigpi)) / ...
                       (1 - temp2);
    
    % Threshold policy
    elseif policy_num == 2
        if tau == 1
            pi = zeros(1,tmax+2);
            pi(1) =  P_rand(2,1) / (p * ET + P_rand(2,1));
            
            pi(2) = (p * (P_rand(2,1) + P_rand(2,2))) / ...
                    (p * ET + P_rand(2,1));
    
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
            theo_penalty = 0;
            for s = 1:tmax
                theo_penalty = theo_penalty + C_rand(s+1) * pi(s+1);
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
            theo_penalty = theo_penalty + ...
                           (temp1 + sum(s_prime .* bigpi)) / (1 - temp2);
    
        else
            omega = tau + tmax;
        
            % Stationary Distribution
            Pi = zeros(omega+2,omega+1);
            
            % First Row (s=0)
            row = 1;
            Pi(row,1) = Pi(row,1) + 1 - p;
            for s = 2:tau
                Pi(row,s) = Pi(row,s) + p;
            end 
            for s = tau+1:omega+1
                Pi(row,s) = Pi(row,s)+ P_rand(1,1);
            end
            
            % Second row (s=1)
            row = 2;
            Pi(row,1) = Pi(row,1) + p;
            for s = tau+1:omega+1
                Pi(row,s) = Pi(row,s) + P_rand(2,2);
            end
            
            % 2 <= s <= tmax-1
            for s = 2:tmax-1
               row = s + 1;
               if s-1 < tau
                   Pi(row,s) = Pi(row,s) + 1 - p;
                   for i = tau:omega
                        Pi(row,i+1) = Pi(row,i+1) + P_rand(tau+1,s+1);
                   end
               end
               if s-1 >= tau
                    for i = tau:s-1
                        Pi(row,i+1) = Pi(row,i+1) + P_rand(i+1,s+1);
                    end
                    for i = s:omega
                       Pi(row,i+1) = Pi(row,i+1) + P_rand(s+1,s+1);
                    end
               end
            end
            
            % s = tmax
            row = tmax + 1;
            if tmax - 1 < tau
                Pi(row,row-1) = Pi(row,row-1) + 1 - p; 
            else
                for i = tau : tmax - 1
                   Pi(row,i+1) = Pi(row,i+1) + P_rand(i+1,tmax+1); 
                end
            end
            
            % tmax + 1 <= s <= omega - 1
            for s = tmax + 1 : omega - 1
                row = s + 1;
                if s - 1 < tau
                   Pi(row,row-1) = Pi(row,row-1) + 1-p; 
                else
                    for i = tau : s-1
                        Pi(row,i+1) = Pi(row,i+1) + P_rand(i+1,s+1);
                    end
                end
            end
            
            % s = omega
            row = omega + 1;
            for s = tau : omega - 1
                temp = 0;
                for k = tau:s
                    temp = temp + P_rand(s+1,tmax+k+1);
                end
                Pi(row,s+1) = Pi(row,s+1) + temp;
            end
            temp = 0;
            for s = 1:tmax
                temp = temp + P_rand(omega+1,omega+s+1);
            end
            Pi(row,row) = Pi(row,row) + temp;
            
            % Together
            together = ones(1,omega+1);
            together(tau+1:omega+1) = ET;
            Pi(end,:) = together;
            
            for s = 0:omega
                Pi(s+1,s+1) =  Pi(s+1,s+1) - 1;
            end
            
            B = zeros(omega+1,1);
            B(end) = 1;
            
            pi = transpose(linsolve(Pi(2:end,:),B));

            % expected AoII
            theo_penalty = 0;
            for s = 1:tau-1
                theo_penalty = theo_penalty + s * pi(s+1);
            end
            for s = tau:omega-1
                theo_penalty = theo_penalty + C_rand(s+1) * pi(s+1);
            end
            temp1 = 0;
            temp2 = 0;
            for t = 1:tmax
                temp3 = 0;
                for s = omega-t:omega-1
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
                for s = omega-t:omega-1
                    temp = temp + pi(s+1);
                end
                temp = temp + pi(end);
                bigpi(t) = pt(t) * P(2,t+1,t) * temp;
            end
            theo_penalty = theo_penalty + ...
                           (temp1 + sum(s_prime .* bigpi)) / (1 - temp2);
        end
    else
        fprintf('Invalid policy index.');
    end
end

%% Function - InstantCost
% Calculate the matrix for the instant cost
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   s_init - AoII starting value
%   tmax - Update transmission time
%   p - Source process dynamics
%   P_chain - State transition probability matrix of the source
% Output:
%   C - the instant cost matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = InstantCost(s_init, tmax, p, P_chain)
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

%% Function - PerformanceSim
% Run the simulations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   epoch - Epoches in each run
%   run - Number of runs
%   thre - Threhsold
%   policy_num - Policy index (1-ZeroWait; 2-Threshold; 3-Lazy)
%   p - Source process dynamics
%   tmax - Maximum transmission time
%   Pr - Probabilities in system dynamic
%   tau - Threshold in Threshold policy (>0)
% Output:
%   sim_AoII - Age of Incorrect Information
%   sim_dist - Stationary distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sim_AoII,sim_dist] = PerformanceSim(epoch, run, thre, ...
                                              policy_num, p, tmax, Pr, tau)
    penalty_out_run = 0;
    age_dist_out_run = 0;
    
    for i = 1:run
        age = 0;
        age_pool = zeros(1,tmax+tau+1);
        penalty_in_run = 0;
        est = 0;           % Estimate at the reciever.
        source = 0;        % State of the source.
        transmitting = 0;  % Current transmitting update
        time_elapsed = 0;      % Transmission time
        for j = 1:epoch
            % Policy
            a = policy(age,thre,time_elapsed,policy_num);
            % Evolve
            [est_,source_,time_elapsed_,transmitting_] = ...
                Evolve(a,est,source,p,time_elapsed,transmitting,Pr);
            if est_ == source_
                age = 0;
            else
                age = age + 1;
            end
            % one step forward
            est = est_;
            source = source_;
            transmitting = transmitting_;
            time_elapsed = time_elapsed_;
            if time_elapsed == 0
                if age > tmax + tau - 1 % For Zero wait policy, tau = 1
                    idx = tmax + tau + 1;
                else
                    idx = age + 1;
                end
                age_pool(idx) = age_pool(idx) + 1;
            end
            penalty_in_run = penalty_in_run + age;
        end
        penalty_out_run = penalty_out_run + penalty_in_run / epoch;
        age_dist_out_run = age_dist_out_run + age_pool ./ epoch;
    end
    sim_AoII = penalty_out_run / run;
    sim_dist = age_dist_out_run ./ run;
end

%% Function - policy
% Return the action according to some policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   age - Current AoII
%   thre - Threshold
%   time_elapsed - The time that the transmission has taken place
%   policy_num - Policy index
% Output:
%   a - Action
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a = policy(age, thre, time_elapsed, policy_num)
    switch policy_num
        case 1 % Zero Wait
            if time_elapsed == 0
                a = 1;
            else
                a = 0;
            end
        case 2 % Threshold
            if age >= thre && time_elapsed == 0
                a = 1;
            else
                a = 0;
            end
        case 3 % Lazy
            a = 0;
        otherwise
            error('Invalid Policy Number!')
    end
end

%% Function - Evolve
% Return the next syatem state for the given current state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   a - Action
%   cur_est - Current estimation
%   cur_source - Cuurent source process state
%   p - Source process dynamics
%   time_elapsed - The time that the transmission has taken place
%   cur_transmitting - The update currently in transmission
%   Pr - Probabilities in systwm dynamic
% Output:
%   nxt_est - Next estimate
%   nxt_source - Next source state
%   nxt_time_elapsed - Next time spent in transmission
%   nxt_transmitting - Next transmitting update
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nxt_est,nxt_source,nxt_time_elapsed,nxt_transmitting] = ...
    Evolve(a, cur_est, cur_source, p, time_elapsed, cur_transmitting, Pr)
    % Source Dynamic
    r_state = binornd(1,p);
    if r_state == 0
        nxt_source = cur_source;
    else
        nxt_source = abs(cur_source-1);
    end
    
    if time_elapsed == 0 % No update transmitting
        if a == 0 % Saty idle
            nxt_est = cur_est;
            nxt_time_elapsed = time_elapsed;
            nxt_transmitting = cur_transmitting;
        else % Transmit
            flag = binornd(1,1-Pr(1));
            if flag == 1 % arrive
                nxt_est = cur_source;
                nxt_time_elapsed = 0;
                nxt_transmitting = cur_source;
            else
                nxt_est = cur_est;
                nxt_time_elapsed = 1;
                nxt_transmitting = cur_source;
            end
        end
    else % Uptdae transmitting
        flag = binornd(1,1-Pr(time_elapsed+1));
        if flag == 1 % arrive
            nxt_est = cur_transmitting;
            nxt_transmitting = cur_transmitting;
            nxt_time_elapsed = 0;
        else
            nxt_est = cur_est;
            nxt_transmitting = cur_transmitting;
            nxt_time_elapsed = time_elapsed + 1;
        end
    end
end
