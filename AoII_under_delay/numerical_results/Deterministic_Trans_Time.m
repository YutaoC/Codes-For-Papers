%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file generates the numerical results as presented in the paper in
% the case where the transmission time is deterministic. See Section V.B.
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% Store result
AoII = zeros(3,11,5);

%% Main Loop
for y = 1:5
    trans_time = y * 2;
    for z = 2:11
        % System Parameters
        p = (z - 1) * 0.05;  % State Transition Probability in Markov Chain
        P_chain = [(1-p),p;p,(1-p)];
        
        policy_num = 2; % Policy index (1-ZeroWait; 2-Threshold; 3-Lazy)
        thre = 1;       % Threshold in Threshold policy (>0)
        
        run = 15;       % Number of runs
        epoch = 15000;  % Epochs in each run
        smax = 300;     % Approximation on number of states
        
        % Renaming due to combining multiple files
        tmax = trans_time;
        tau = thre;
        
        % Instant Costs
        C_trans = zeros(smax+1,1);   % Cost for transmission
        C_notrans = zeros(smax+1,1); % Cost for staying idle
        for i = 0:smax
            C_trans(i+1) = InstantCost(i,tmax,p,P_chain);
            C_notrans(i+1) = i;
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
        
        % Simulation
        sim_AoII = PerformanceSim(trans_time, epoch, run, thre, ...
                                  policy_num, p);
        AoII(1,z,y) = sim_AoII;
        
        % Theoretical
        [theo_AoII,~] = PerformanceTheo(p,tmax,tau,P,C_trans, ...
                                        C_notrans,policy_num);
        AoII(2,z,y) = theo_AoII;
        
        % Accuracy
        AoII(3,z,y) = 100 * abs(sim_AoII-theo_AoII) / ...
                      mean([sim_AoII,theo_AoII]);
    end
end

%% Store the results
save('AoII.mat', 'AoII');

%% Function - PerformanceSim
% Run the simulations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   trans_time - Update transmission time
%   epoch - Epoches in each run
%   run - Number of runs
%   thre - Threhsold
%   policy_num - Policy index (1-ZeroWait; 2-Threshold; 3-Lazy)
%   p - Source process dynamics
% Output:
%   AoII - expected Age of Incorrect Information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function AoII = PerformanceSim(trans_time, epoch, run, thre, policy_num, p)
    penalty_out_run = 0;
    for i = 1:run
        age = 0;
        penalty_in_run = 0;
        est = 0;                    % Estimate at the reciever.
        source = 0;                 % State of the source.
        transmitting = 0;           % The transmitting update
        time_elapsed = trans_time;  % Transmission time
        for j = 1:epoch
            % Policy
            a = policy(age,thre,time_elapsed,trans_time,policy_num);
            % Evolve
            [est_,source_,time_elapsed_,transmitting_] = Evolve(a,est, ...
                source,p,time_elapsed,trans_time,transmitting);
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
            penalty_in_run = penalty_in_run + age;
        end
        penalty_out_run = penalty_out_run + penalty_in_run / epoch;
    end
    AoII = penalty_out_run / run;
end

%% Function - Theoretical
% Calculate the theoretical value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   p - Source process dynamics
%   tmax - update transmission time
%   tau - Threshold
%   P - State transition probabilities
%   C_trans - Cost when transmission happens
%   C_notrans - Cost when staying idle
% Output:
%   AoII - Expected Age of Incorrect Information
%   dist - The stationary distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [AoII,dist] = PerformanceTheo(p,tmax,tau,P,C_trans,C_notrans, ...
                                       policy_num)
    % Lazy policy
    if policy_num == 3
        AoII = 1 / (2 * p);
        dist = 0;
    
    % Zero Wait policy
    elseif policy_num == 1
        pi = zeros(1,tmax+2);
        pi(1) =  P(2,1) / tmax;
        for i = 2:tmax
            pi(i) =  (P(2,i)+P(2,1)*(P(1,i)-P(2,i))) / tmax;
        end
        pi(tmax+1) = P(1,end) * pi(1);
        pi(tmax+2) = P(2,end) * ((1-P(2,1))/tmax);

        dist = pi;
        
        % The expected AoII
        numerator = tmax * ((1-(1-p)^tmax) / p) * pi(end);
        for s = 2:tmax+1
            numerator = numerator + C_trans(s) * pi(s);
        end
        denominator = 1 - P(2,end);
        AoII = C_trans(1) * pi(1) + numerator / denominator;

    % Threshold policy
    elseif policy_num == 2
        if tau == 1
            dist = zeros(1,tmax+2);
            dist(2) = ((P(2,1)+P(2,2))*p) / (P(2,1) + p*tmax);
            for s = 2:tmax-1
                dist(s+1) = (P(2,s+1)*p) / (P(2,1)+p*tmax);
            end
            dist(end) = (P(2,end)*p) / (P(2,1)+p*tmax);
            AoII = 0;
            for s = 1:tmax
                AoII = AoII + C_trans(s+1) * dist(s+1);
            end
            AoII = AoII / (1-P(2,end));
            AoII = AoII + ((tmax - tmax*(1-p)^tmax) / (p-p*P(2,end))) * ...
                   dist(end);
        else
            omega = tau + tmax;
    
            % Stationary Distribution
            Pi = zeros(omega+2,omega+1);
            
            % First Row (s=0)
            row = 1;
            Pi(row,1) = 1 - p;
            for s = 2:tau
                Pi(row,s) = p;
            end 
            for s = tau+1:omega
                Pi(row,s) = P(2,1);
            end
            Pi(row,omega+1) = P(2,1);
            
            % Second row (s=1)
            row = 2;
            Pi(row,1) = p;
            for s = tau+1:omega
                Pi(row,s) = P(2,2);
            end
            Pi(row,omega+1) = P(2,2);
            
            % 2 <= s <= tmax-1
            for s = 2:tmax-1
               row = s + 1;
               if s-1 < tau
                   Pi(row,s) = 1 - p;
               end
               for i = tau+1:omega
                   Pi(row,i) = P(2,s + 1);
               end
               Pi(row,omega+1) = P(2,s + 1);
            end
            
            % s = tmax
            row = tmax + 1;
            if tmax - 1 < tau
               Pi(row,tmax) = 1 - p; 
            end
            
            % tmax + 1 <= s <= omega - 1
            for s = tmax + 1 : omega - 1
                row = s + 1;
                if s - 1 < tau
                   Pi(row,row-1) = 1-p; 
                elseif s - tmax >= tau
                    Pi(row,row-tmax) = P(2,tmax + 1);
                end
            end
            
            % s = omega
            s = omega;
            row = s + 1;
            for s = tau + 1 : omega
                Pi(row,s) = P(2,tmax+1);
            end
            Pi(row,row) = P(2,tmax+1);
            
            % Together
            together = ones(1,omega+1);
            together(tau+1:omega+1) = tmax;
            Pi(end,:) = together;
            
            for s = 0:omega
                Pi(s+1,s+1) =  Pi(s+1,s+1) - 1;
            end
            
            B = zeros(omega+1,1);
            B(end) = 1;
            
            dist = linsolve(Pi(2:end,:),B);
            
            % Expected AoII
            Expected = zeros(omega+1,1);
            for i = 1 : tau
               Expected(i) = C_notrans(i) * dist(i); 
            end
            
            for i = tau+1:omega
               Expected(i) = C_trans(i) * dist(i); 
            end
            % SIGMA
            for i = tau+1 : omega
                Expected(omega+1) = Expected(omega+1) + ...
                                    P(2,tmax+1) * C_trans(i)*dist(i); 
            end
            Expected(omega+1) = Expected(omega+1) + ...
                                tmax * ((1-(1-p)^tmax)/p) *dist(end);
            Expected(omega+1) = Expected(omega+1) / (1-P(2,tmax+1));
            
            AoII = sum(Expected);
        end
    else
        fprintf('Invalid policy index.');
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
%   trans_time - Update transmission time
%   cur_transmitting - The update currently in transmission
% Output:
%   nxt_est - Next estimate
%   nxt_source - Next source state
%   nxt_time_elapsed - Next time spent in transmission
%   nxt_transmitting - Next transmitting update
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nxt_est,nxt_source,nxt_time_elapsed,nxt_transmitting] = ...
    Evolve(a,cur_est,cur_source,p,time_elapsed,trans_time,cur_transmitting)
    % Source Dynamic
    r_state = binornd(1,p);
    if r_state == 0
        nxt_source = cur_source;
    else
        nxt_source = abs(cur_source-1);
    end
    
    if time_elapsed == trans_time % No update transmitting
        nxt_est = cur_est;
        if a == 0
            nxt_time_elapsed = time_elapsed;
            nxt_transmitting = cur_transmitting;
        else
            nxt_time_elapsed = 1;
            nxt_transmitting = cur_source;
        end
    else % Uptdae transmitting
        nxt_time_elapsed = time_elapsed + 1;
        if nxt_time_elapsed == trans_time % arrive
            nxt_est = cur_transmitting;
            nxt_transmitting = cur_transmitting;
        else
            nxt_est = cur_est;
            nxt_transmitting = cur_transmitting;
        end
    end
end

%% Function - policy
% Return the action according to some policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   age - Current AoII
%   thre - Threshold
%   time_elapsed - The time that the transmission has taken place
%   trans_time - Update transmission time
%   policy_num - Policy index
% Output:
%   a - Action
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a = policy(age,thre,time_elapsed,trans_time,policy_num)
    switch policy_num
        case 1 % Zero Wait
            if time_elapsed == trans_time
                a = 1;
            else
                a = 0;
            end
        case 2 % Threshold
            if age >= thre && time_elapsed == trans_time
                a = 1;
            else
                a = 0;
            end
        case 3 % Lazy
            a = 0;
        otherwise
            error('Invalid Policy Number')
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
