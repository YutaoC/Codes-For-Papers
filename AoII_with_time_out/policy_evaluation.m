%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file calculates the expected AoII of the system with time out 
% resulting from the adoption of various policies - Both theoretical & 
% simulation results.
% Author: Yutao Chen
% Updated: 07/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% System Parameters
p = 0.35;    % State Transition Probability in Markov Chain
tmax = 2;   % Maximum transmission time
P_chain = [(1-p),p;p,(1-p)];

% Policy parameters (Change tau according to policy for simulation)
policy_num = 2; % Policy index (1-ZeroWait; 2-Threshold; 3-Lazy)
thre = 4;       % Threshold in Threshold policy (>0)
tau = thre;     % Renaming due to combination of multiple files

% Performance simulation parameters
run = 15;      % Number of runs
epoch = 25000; % Epochs in each run

smax = 300; % Calculate quantities for only a finite number of states

%% Transmission time distribution
pb = 0.6;   % (Bernoulli) successful probability
pt = zeros(1,tmax+1);
for i = 1:tmax
    pt(i) = (1-pb)^(i-1) * pb;
end
pt(tmax+1) = 1 - sum(pt(1:tmax));

%% Derived variables
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

%% Policy Performance Evaluation - Simulation
fprintf('...Simulating policy %d\n', policy_num);

[sim_AoII, sim_dist] = PerformanceSim(epoch, run, thre, policy_num, p, ...
                                      tmax, Pr, tau);

fprintf('(Sim) Expected AoII = %.4f \n', sim_AoII);
fprintf('\n');

%% Policy Evaluation - Theoretical calculation
fprintf('...Calculating policy %d\n', policy_num);

[theo_AoII,theo_dist,~] = PerformanceTheo(p, P_rand, tmax, tau, ET, ...
                                          policy_num, C_rand, pt, U);

fprintf('(Theo) Expected AoII = %.4f \n', theo_AoII);
fprintf('\n');

%% Accuracy check
fprintf('Difference = %.4f \n', abs(sim_AoII - theo_AoII));
fprintf('Percentage = %.4f %%\n', 100 * abs(sim_AoII - theo_AoII) / ...
                                  mean([sim_AoII,theo_AoII]));
fprintf('\n');

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
        time_elapsed = 0;  % Transmission time
        for j = 1:epoch
            % Policy
            a = policy(age,thre,time_elapsed,policy_num);
            % Evolve
            [est_,source_,time_elapsed_,transmitting_] = Evolve(a,est, ...
                source,p,time_elapsed,transmitting,Pr,tmax);
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
                if age >= tmax + tau % For Zero wait policy, tau = 0
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
    Evolve(a, cur_est, cur_source, p, time_elapsed, cur_transmitting, Pr,tmax)
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
                if tmax == 1
                    nxt_est = cur_est;
                    nxt_time_elapsed = 0;
                    nxt_transmitting = cur_transmitting;
                else
                    nxt_est = cur_est;
                    nxt_time_elapsed = 1;
                    nxt_transmitting = cur_source;
                end
            end
        end
    else % Uptdae transmitting
        flag = binornd(1,1-Pr(time_elapsed+1));
        if flag == 1 % arrive
            nxt_est = cur_transmitting;
            nxt_transmitting = cur_transmitting;
            nxt_time_elapsed = 0;
        else
            if time_elapsed == tmax - 1
                nxt_est = cur_est;
                nxt_transmitting = cur_transmitting;
                nxt_time_elapsed = 0;
            else
                nxt_est = cur_est;
                nxt_transmitting = cur_transmitting;
                nxt_time_elapsed = time_elapsed + 1;
            end
        end
    end
end
