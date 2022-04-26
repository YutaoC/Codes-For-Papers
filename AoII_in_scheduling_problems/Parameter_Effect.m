%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file calculates and evaluates the following scheduling policies.
% 1. Whittle's Index policy   (WIP)
% 2. Indexed Priority policy  (IPP)
% 3. Benchmark Performance    (BP)
% 4. Greedy/Greedy+ Policy    (G/G+P)
% Notes: Gamma are set to be the same across all users.
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% System Parameters
N = 15;      % Number of users in the system
smax = 800;  % Truncate the value of s

% Weights in time penalty funtions
% w = linspace(0.5,1.5,N);
w = linspace(1,1,N);

% Probabilities for the source to change states
p_pool = linspace(0.05,0.45,N);
% p_pool = 0.3 * linspace(1,1,N);

% Probability of error when CSI indicates failure (0)
% pe0_pool = linspace(0,0.45,N);
pe0_pool = 0.1 * linspace(1,1,N);
% pe0_pool = 0 * linspace(1,1,N); 

% Probability of error when CSI indicates success (1)
% pe1_pool = linspace(0,0.45,N);
pe1_pool = 0.1 * linspace(1,1,N); 

% Probability distribution of CSI
% gamma_pool = linspace(0,0.45,N);
gamma_pool = 0.6 * linspace(1,1,N);

% Auxiliarily quantites (see paper for definition)
alpha_pool = pe1_pool.*(1-p_pool) + (1-pe1_pool).*p_pool;
beta_pool = pe0_pool.*p_pool + (1-pe0_pool).*(1-p_pool);
c1_pool = (1-gamma_pool).*(1-p_pool) + gamma_pool.*alpha_pool;
c2_pool = (1-gamma_pool).*beta_pool + gamma_pool.*alpha_pool;

% Store the results
AoII_ALL = zeros(1,5);

%% Whittle's Index
Whittle = zeros(smax/2,N);
for j = 1:N
    p = p_pool(j);
    pe1 = pe1_pool(j);
    gamma = gamma_pool(j);
    alpha = alpha_pool(j);
    c1 = c1_pool(j);
    const = ((1-c1)*(1-p)-gamma*(1-p-alpha)) / (c1*(1-p-alpha));
    for i = 1:smax/2
        [AoII,rate] = expected_value_reduced(p,c1,i,gamma,smax,j,w);
        tmp = 0;
        for k = i+1:smax
           tmp = tmp + f(k,j,w) * c1^(k-i-1); 
        end
        Whittle(i,j) = ((1-c1)*tmp - AoII)/(const + rate);
    end
end

%% Policy Evaluation - WIP
run = 15;       % Number of runs
epoch = 15000;  % Epochs in each run

AoII_in_run = 0;

for i = 1:run
    cumulative_AoII = 0;
    ages = zeros(1,N);
    C = binornd(1,gamma,[1,N]); % CSI
    for j = 1:epoch
        % Policy
        a = WIP(Whittle,ages,C,N);
        % Evolve
        ages_ = NextStep(a,ages,pe1_pool,pe0_pool,C,p_pool,N);
        C = binornd(1,gamma,[1,N]); % CSI
        cumulative_AoII = cumulative_AoII + f_all(ages,N,w);
        % one step forward
        ages = ages_;
    end
    AoII_in_run = AoII_in_run + cumulative_AoII / epoch;
end
AoII = AoII_in_run / run;

AoII_ALL(1) = AoII;

fprintf('------------------------------------\n');
fprintf('N = %d \n', N);
fprintf('(WIP) Average AoII = %.4f \n', AoII/N);
fprintf('------------------------------------\n');

%% Optimal policy for RP
% Returns two deterministic policies and corresponding value functions
epsilon = 0.01;      % Convergence criteria in RVI
n_est = zeros(2,N);  % Threshold vector
rho = zeros(1,N);    % Transmission rate

% Initalization
lambda_low = 0;
lambda_high = 1;

for j = 1:N
    p = p_pool(j);
    c1 = c1_pool(j);
    c2 = c2_pool(j);
    pe1 = pe1_pool(j);
    pe0 = pe0_pool(j);
    gamma = gamma_pool(j);
    [n_est(:,j),~] = RVI(lambda_high,p,gamma,pe1,pe0,smax,epsilon,j,w);
    [~,rho(j)] = expected_value(p,c1,c2,n_est(:,j),gamma,smax,j,w);
end
Rate = sum(rho);

% Find the range of lambda
while Rate >= 1
    lambda_low = lambda_high;
    lambda_high = lambda_low * 2;
    for j = 1:N
        p = p_pool(j);
        c1 = c1_pool(j);
        c2 = c2_pool(j);
        pe1 = pe1_pool(j);
        pe0 = pe0_pool(j);
        gamma = gamma_pool(j);
        [n_est(:,j),~] = RVI(lambda_high,p,gamma,pe1,pe0,smax,epsilon,j,w);
        [~,rho(j)] = expected_value(p,c1,c2,n_est(:,j),gamma,smax,j,w);
    end
    Rate = sum(rho);
end

% Binary search
while abs(lambda_high - lambda_low) > 0.001
    lambda = (lambda_low + lambda_high) / 2;
    for j = 1:N
        p = p_pool(j);
        c1 = c1_pool(j);
        c2 = c2_pool(j);
        pe1 = pe1_pool(j);
        pe0 = pe0_pool(j);
        gamma = gamma_pool(j);
        [n_est(:,j),~] = RVI(lambda,p,gamma,pe1,pe0,smax,epsilon,j,w);
        [~,rho(j)] = expected_value(p,c1,c2,n_est(:,j),gamma,smax,j,w);
    end
    Rate = sum(rho);
    if Rate >= 1
       lambda_low = lambda;
    else
       lambda_high = lambda;
    end
end

% Lambda+
VF_plus = zeros(smax+1,2,N);  % Corresponding value function
n_plus = zeros(2,N);          % Corresponding threshold vector
for j = 1:N
    p = p_pool(j);
    pe1 = pe1_pool(j);
    pe0 = pe0_pool(j);
    gamma = gamma_pool(j);
    [n_plus(:,j),VF_plus(:,:,j)] = RVI(lambda_high,p,gamma,pe1,pe0, ...
                                       smax,epsilon,j,w);
end

% Lambda-
VF_minus = zeros(smax+1,2,N);  % Corresponding value function
n_minus = zeros(2,N);          % Corresponding threshold vector
for j = 1:N
    p = p_pool(j);
    pe1 = pe1_pool(j);
    pe0 = pe0_pool(j);
    gamma = gamma_pool(j);
    [n_minus(:,j),VF_minus(:,:,j)] = RVI(lambda_low,p,gamma,pe1,pe0, ...
                                         smax,epsilon,j,w);
end

%% Benchmark Performance - BP
if isequal(n_plus,n_minus)
    AoII = 0;
    for j = 1:N
        p = p_pool(j);
        c1 = c1_pool(j);
        c2 = c2_pool(j);
        gamma = gamma_pool(j);
        [tmp,~] = expected_value(p,c1,c2,n_plus(:,j),gamma,smax,j,w);
        AoII = AoII + tmp;
    end
else
    Pbar_plus = 0;
    Pbar_minus = 0;
    Cbar_plus = 0;
    Cbar_minus = 0;
    for j = 1:N
        p = p_pool(j);
        c1 = c1_pool(j);
        c2 = c2_pool(j);
        gamma = gamma_pool(j);
        [tmp_a1,tmp_r1] = expected_value(p,c1,c2,n_plus(:,j), ...
                                         gamma,smax,j,w);
        [tmp_a2,tmp_r2] = expected_value(p,c1,c2,n_minus(:,j), ...
                                         gamma,smax,j,w);
        Pbar_plus = Pbar_plus + tmp_r1;
        Pbar_minus = Pbar_minus + tmp_r2;
        Cbar_plus = Cbar_plus + tmp_a1;
        Cbar_minus = Cbar_minus + tmp_a2;
    end
    mu = (1 - Pbar_plus) / (Pbar_minus - Pbar_plus);
    AoII = mu * Cbar_minus + (1 - mu) * Cbar_plus;
end

AoII_ALL(2) = AoII;

fprintf('------------------------------------\n');
fprintf('N = %d \n', N);
fprintf('(BP) Average AoII = %.4f \n', AoII/N);
fprintf('------------------------------------\n');

%% Index Ix
Ix = zeros(smax/2,2,N);
for j = 1:N
    p = p_pool(j);
    pe1 = pe1_pool(j);
    pe0 = pe0_pool(j);
    gamma = gamma_pool(j);
    for i = 1:smax/2
        for r = 1:2
            Trans1 = RVI_Trans(i,r,p,gamma,pe1,pe0,smax,VF_plus(:,:,j),1);
            Trans2 = RVI_Trans(i,r,p,gamma,pe1,pe0,smax,VF_plus(:,:,j),2);
            Ix(i,r,j) = Trans1-Trans2-lambda_high;
        end
    end
end

%% Policy Evaluation - IPP
run = 15;       % Number of runs
epoch = 15000;  % Epochs in each run

AoII_in_run = 0;

for i = 1:run
    cumulative_AoII = 0;
    ages = zeros(1,N);
    C = binornd(1,gamma_pool(1),[1,N]); % CSI
    for j = 1:epoch
        % Policy
        a = IPP(Ix,ages,C,N);
        % Evolve
        ages_ = NextStep(a,ages,pe1_pool,pe0_pool,C,p_pool,N);
        C = binornd(1,gamma_pool(1),[1,N]); % CSI
        cumulative_AoII = cumulative_AoII + f_all(ages,N,w);
        % one step forward
        ages = ages_;
    end
    AoII_in_run = AoII_in_run + cumulative_AoII / epoch;
end
AoII = AoII_in_run / run;

AoII_ALL(3) = AoII;

fprintf('------------------------------------\n');
fprintf('N = %d \n', N);
fprintf('(IPP) Average AoII = %.4f \n', AoII/N);
fprintf('------------------------------------\n');

%% Policy Evaluation - G+P
run = 15;       % Number of runs
epoch = 15000;  % Epochs in each run

AoII_in_run = 0;

for i = 1:run
    cumulative_AoII = 0;
    ages = zeros(1,N);
    C = binornd(1,gamma_pool(1),[1,N]); % CSI
    for j = 1:epoch
        % Policy
        a = Greedy_plus(ages,N,C);
        % Evolve
        ages_ = NextStep(a,ages,pe1_pool,pe0_pool,C,p_pool,N);
        C = binornd(1,gamma_pool(1),[1,N]); % CSI
        cumulative_AoII = cumulative_AoII + f_all(ages,N,w);
        % one step forward
        ages = ages_;
    end
    AoII_in_run = AoII_in_run + cumulative_AoII / epoch;
end
AoII = AoII_in_run / run;

AoII_ALL(4) = AoII;

fprintf('------------------------------------\n');
fprintf('N = %d \n', N);
fprintf('(G+P) Average AoII = %.4f \n', AoII/N);
fprintf('------------------------------------\n');

%% Policy Evaluation - GP
run = 15;       % Number of runs
epoch = 15000;  % Epochs in each run

AoII_in_run = 0;

for i = 1:run
    cumulative_AoII = 0;
    ages = zeros(1,N);
    C = binornd(1,gamma_pool(1),[1,N]); % CSI
    for j = 1:epoch
        % Policy
        a = Greedy(ages,N);
        % Evolve
        ages_ = NextStep(a,ages,pe1_pool,pe0_pool,C,p_pool,N);
        C = binornd(1,gamma_pool(1),[1,N]); % CSI
        cumulative_AoII = cumulative_AoII + f_all(ages,N,w);
        % one step forward
        ages = ages_;
    end
    AoII_in_run = AoII_in_run + cumulative_AoII / epoch;
end
AoII = AoII_in_run / run;

AoII_ALL(5) = AoII;

fprintf('------------------------------------\n');
fprintf('N = %d \n', N);
fprintf('(GP) Average AoII = %.4f \n', AoII/N);
fprintf('------------------------------------\n');

%% Function - expected_value
% Follows Proposition 2 of the paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   p - Probabilities for the source to change states
%   c1 - See paper for definition
%   c2 - See paper for definition
%   n - Threshold
%   gamma - Probability distribution of CSI
%   smax - Truncate the value of s
%   j - User index
%   w - Weights in time penalty funtions
% Output:
%   AoII - Expected Age of Incorrect Information
%   rate - Expected rate of transmission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [AoII,rate] = expected_value(p,c1,c2,n,gamma,smax,j,w)
    if n(1) == smax + 2
        [AoII,rate] = expected_value_reduced(p,c1,n(2),gamma,smax,j,w);
    else
        n0 = n(1);
        n1 = n(2);

        pi0_rev = 2 + p * (1-p)^(n1-1)*(1/(1-c1) - 1/p + c1^(n0-n1)*(1/(1-c2)-1/(1-c1)));

        rate_tmp = p * (1-p)^(n1-1) * (gamma/(1-c1) + c1^(n0-n1)*(1/(1-c2)-gamma/(1-c1)));
        rate = rate_tmp / pi0_rev;

        tmp1 = 0;
        for k = 1:n1-1
           tmp1 = tmp1 + f(k,j,w)*(1-p)^(k-1) ;
        end
        tmp2 = 0;
        for k = n1:n0-1
           tmp2 = tmp2 + f(k,j,w) * c1^(k-n1); 
        end
        tmp3 = 0;
        for k = n0:smax
           tmp3 = tmp3 + f(k,j,w) * c2^(k-n0); 
        end

        AoII_tmp = p * tmp1 + p * (1-p)^(n1-1) * (tmp2 + c1^(n0-n1) * tmp3);
        AoII = AoII_tmp / pi0_rev;
    end
end

%% Function - expected_value_reduced
% Follows Corollary 5 of the paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   p - Probabilities for the source to change states
%   c1 - See paper for definition
%   n - Threshold
%   gamma - Probability distribution of CSI
%   smax - Truncate the value of s
%   j - User index
%   w - Weights in time penalty funtions
% Output:
%   AoII - Expected Age of Incorrect Information
%   rate - Expected rate of transmission
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [AoII,rate] = expected_value_reduced(p,c1,n,gamma,smax,j,w)
    pi0_rev = 2 + p * (1-p)^(n-1)*(1/(1-c1) - 1/p);

    rate_tmp = p * (1-p)^(n-1) * (gamma/(1-c1));
    rate = rate_tmp / pi0_rev;

    tmp1 = 0;
    for k = 1:n-1
       tmp1 = tmp1 + f(k,j,w)*(1-p)^(k-1) ;
    end
    tmp2 = 0;
    for k = n:smax
       tmp2 = tmp2 + f(k,j,w) * c1^(k-n); 
    end

    AoII_tmp = p * tmp1 + p * (1-p)^(n-1) * tmp2;
    AoII = AoII_tmp / pi0_rev;
end

%% Function - f_all
% Applying the time penalty function to the penalty to get AoII (All user)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   s - Penalties for each user
%   N - Number of users in the system
%   w - Weights in time penalty funtions
% Output:
%   ret - AoII for each user
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ret = f_all(s,N,w)
    ret = 0;
    for i = 1:N
       ret = ret + f(s(i),i,w);
    end
end

%% Function - f
% Applying the time penalty function to the penalty to get AoII (one user)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   s - Penalty
%   N - User index
%   w - Weight in time penalty funtion
% Output:
%   ret - AoII for this user only
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ret = f(s,i,w)
    ret = s^w(i);
end

%% Function - RVI
% Return the optimal policy and the value functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   lambda - Lagrange multiplier
%   p - Probabilities for the source to change states
%   gamma - Probability distribution of CSI
%   pe1 - Probability of error when CSI indicates success (1)
%   pe0 - Probability of error when CSI indicates failure (0)
%   smax - Truncate the value of s
%   epsilon - Convergence criteria in RVI
%   j - User index
%   w - Weights in time penalty funtions
% Output:
%   ret_policy - The optimal policy
%   ret_VF - The value functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ret_policy,ret_VF] = RVI(lambda,p,gamma,pe1,pe0,smax,epsilon,j,w)
    RV_old = zeros(smax+1,2);    % Value function
    action_R = zeros(smax+1,2);  % Resulting actions
    cnt = 0;                     % Counter
    iter_max = 10000;            % Maximum iteration
    H = zeros(smax+1,2);         % Interim value function

    % Instant cost
    C = zeros(smax+1,2);
    for i = 1:smax+1
        for a = 1:2
           C(i,a) = f(i-1,j,w) + (a-1) * lambda;
        end
    end

    % Main Loop
    while cnt <= iter_max
        for i = 1:smax+1
            for j = 1:2
                Trans1 = RVI_Trans(i,j,p,gamma,pe1,pe0,smax,RV_old,1);
                Trans2 = RVI_Trans(i,j,p,gamma,pe1,pe0,smax,RV_old,2);
                H(i,j) = min([Trans1+C(i,1),Trans2+C(i,2)]);
            end
        end
        RV_new = H - H(1,1);

        % Generate the threshold vector
        if max(abs(RV_old - RV_new)) <= epsilon
            for i = 1:smax+1
                for j = 1:2
                    Trans1 = RVI_Trans(i,j,p,gamma,pe1,pe0,smax,RV_old,1);
                    Trans2 = RVI_Trans(i,j,p,gamma,pe1,pe0,smax,RV_old,2);
                    [~,action_R(i,j)] = min([Trans1+C(i,1),Trans2+C(i,2)]);
                end
            end
            break
        end
        % One step forward
        RV_old = RV_new;
        cnt = cnt + 1;
    end
    ret_VF = RV_new;
    if isempty(find(action_R(:,1)>1,1))
        n0 = smax + 2;
    else
        n0 = max(1, find(action_R(:,1)>1,1) - 1);
    end
    n1 = max(1, find(action_R(:,2)>1,1) - 1);
    
    ret_policy = [n0 n1];
end

%% Function - RVI_Trans
% Return the value function under given action
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   i - Penalty
%   j - User index
%   p - Probabilities for the source to change states
%   gamma - Probability distribution of CSI
%   pe1 - Probability of error when CSI indicates success (1)
%   pe0 - Probability of error when CSI indicates failure (0)
%   smax - Truncate the value of s
%   RV_old - Current value function
%   action - The given action
% Output:
%   ret - The resulting value function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ret = RVI_Trans(i,j,p,gamma,pe1,pe0,smax,RV_old,action)
    if i == 1 % s = 0
        ret = (1-gamma)*(1-p)*RV_old(1,1) + (1-gamma)*p*RV_old(2,1) + ...
              gamma*(1-p)*RV_old(1,2) + gamma*p*RV_old(2,2);
    elseif i ~= 1 && j == 1 % negative CSI
        if action == 2 % Transmit
            ret = (1-gamma)*(pe0*(1-p)+(1-pe0)*p)*RV_old(1,1) + ...
                  (1-gamma)*(pe0*p+(1-pe0)*(1-p))*RV_old(min(i+1,smax+1),1) + gamma*(pe0*(1-p)+(1-pe0)*p)*RV_old(1,2) + gamma*(pe0*p+(1-pe0)*(1-p))*RV_old(min(i+1,smax+1),2);
        else
            ret = (1-gamma)*p*RV_old(1,1) + (1-gamma)*(1-p)* ...
                  RV_old(min(i+1,smax+1),1) + gamma*p*RV_old(1,2) + ...
                  gamma*(1-p)*RV_old(min(i+1,smax+1),2);
        end
    else % positive CSI
        if action == 2
            ret = (1-gamma)*(pe1*p+(1-pe1)*(1-p))*RV_old(1,1) + ...
                  (1-gamma)*(pe1*(1-p)+(1-pe1)*p)*RV_old(min(i+1,smax+1),1) + gamma*(pe1*p+(1-pe1)*(1-p))*RV_old(1,2) + gamma*(pe1*(1-p)+(1-pe1)*p)*RV_old(min(i+1,smax+1),2);
        else
            ret = (1-gamma)*p*RV_old(1,1) + (1-gamma)*(1-p)* ...
                  RV_old(min(i+1,smax+1),1) + gamma*p*RV_old(1,2) + ...
                  gamma*(1-p)*RV_old(min(i+1,smax+1),2);
        end
    end
end

%% Function - NextStep
% Return the next states for all users given the current states
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   a - Action
%   cur_age - Current age of all users
%   pe1 - Probability of error when CSI indicates success (1)
%   pe0 - Probability of error when CSI indicates failure (0)
%   C - Channel State Information
%   p_pool - Probabilities for the source to change states
%   N - Number of users in the system
% Output:
%   ret - The next states for all users
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ret = NextStep(a,cur_age,pe1,pe0,C,p_pool,N)
    ret = zeros(1,N);
    for i = 1:N
        ret(i) = Evolve(a(i),cur_age(i),pe0(i),pe1(i),C(i),p_pool(i));
    end
end

%% Function - Evolve
% Return the next state for a single user given the current state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   a - Action
%   cur_age - Current age
%   pe1 - Probability of error when CSI indicates success (1)
%   pe0 - Probability of error when CSI indicates failure (0)
%   u - Channel State Information
%   p - Probability for the source to change states
% Output:
%   ret - The next state for this user
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ret = Evolve(a,cur_age,pe0,pe1,u,p)
    if cur_age == 0
        r = binornd(1,p);  % whether the source changes state
        if r == 1
            ret = 1;
        else
            ret = 0;
        end
    else
        r = binornd(1,p);     % whether the source changes state
        e0 = binornd(1,pe0);  % channel realization when CSI=0
        e1 = binornd(1,pe1);  % channel realization when CSI=1
        if a == 0
            if r == 1
                ret = 0;
            else
                ret = cur_age + 1;
            end
        else
            if u == 0
                if e0 == 1 && r == 1
                    ret = cur_age + 1;
                elseif e0 == 0 && r == 0
                    ret = cur_age + 1;
                else
                    ret = 0;
                end
            else
                if e1 == 1 && r ==0
                    ret = cur_age + 1;
                elseif e1 == 0 && r == 1
                    ret = cur_age + 1;
                else
                    ret = 0;
                end
            end
        end
    end
end

%% Function - WIP
% Return the action according to Whittle's index policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   Whittle - Whittle's index
%   cur_age - Current age
%   C - Channel State Information
%   N - Number of users in the system
% Output:
%   a - The suggested action
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a = WIP(Whittle,cur_age,C,N)
    a = zeros(1,N);
    tmp = zeros(1,N);
    for i = 1:N
        if C(i) == 1 && cur_age(i) > 0
           tmp(i) = Whittle(cur_age(i),i);
        end
    end
    [~,idx] = max(tmp);
    a(idx) = 1;
end

%% Function - IPP
% Return the action according to Indexed priority policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   Whittle - Whittle's index
%   cur_age - Current age
%   C - Channel State Information
%   N - Number of users in the system
% Output:
%   a - The suggested action
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a = IPP(Ix,cur_age,C,N)
    a = zeros(1,N);
    tmp = zeros(1,N);
    for i = 1:N
        tmp(i) = Ix(cur_age(i)+1,C(i)+1,i);
    end
    [~,idx] = max(tmp);
    a(idx) = 1;
end

%% Function - Greedy
% Return the action according to Greedy policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   cur_age - Current age
%   N - Number of users in the system
% Output:
%   a - The suggested action
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a = Greedy(cur_age,N)
    a = zeros(1,N);
    [~,idx] = max(cur_age);
    a(idx) = 1;
end

%% Function - Greedy_plus
% Return the action according to Greedy+ policy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   cur_age - Current age
%   N - Number of users in the system
%   C - Channel State Information
% Output:
%   a - The suggested action
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a = Greedy_plus(cur_age,N,C)
    a = zeros(1,N);
    tmp = zeros(1,N);
    for i = 1:N
        if C(i) == 1
            tmp(i) = cur_age(i);
        end
    end
    [~,idx] = max(tmp);
    a(idx) = 1;
end
