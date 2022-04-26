function [C_WIP,C_IPP,C_BP,C_GP] = run_Whittle(N,w,p_pool,pe0_pool,pe1_pool,gamma_pool)
    alpha_pool = pe1_pool.*(1-p_pool) + (1-pe1_pool).*p_pool;
    beta_pool = pe0_pool.*p_pool + (1-pe0_pool).*(1-p_pool);

    c1_pool = (1-gamma_pool).*(1-p_pool) + gamma_pool.*alpha_pool;
    c2_pool = (1-gamma_pool).*beta_pool + gamma_pool.*alpha_pool;

    smax = 800; % Bounds on the value of s
    
    % Whittle's Index
    % Notes: Exclude s=0
    Whittle = zeros(smax/2,N);
    for j = 1:N
        p = p_pool(j);
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

    % Policy Evaluation - WIP
    run = 15;
    epoch = 15000;
    expected_penalty = 0;
    for i = 1:run
        cumulative_penalty = 0;
        cur_age = zeros(1,N);
%         C = binornd(1,gamma,[1,N]); % CSI % Gamma is set to be the same.
        C = realization(gamma_pool,N);
        for j = 1:epoch
            % Policy
            a = WIP(Whittle,cur_age,C,N);
            % Evolve
            nxt_ages = NextStep(a,cur_age,pe1_pool,pe0_pool,C,p_pool,N); % Evolve
%             C = binornd(1,gamma,[1,N]); % CSI % Gamma is set to be the same.
            C = realization(gamma_pool,N);
            cumulative_penalty = cumulative_penalty + f_all(cur_age,N,w);
            % one step forward
            cur_age = nxt_ages;
        end
        expected_penalty = expected_penalty + cumulative_penalty / epoch;
    end
    C_WIP = expected_penalty / run;

%     fprintf('------------------------------------\n');
%     fprintf('N = %d \n', N);
%     fprintf('(WIP) Sum AoII = %.4f \n', Cbar);
%     fprintf('(WIP) Average AoII = %.4f \n', Cbar/N);
%     fprintf('------------------------------------\n');

    % Optimal policy for RP
    % Returns two deterministic policies and corresponding value functions
    epsilon = 0.01; % Convergence criteria in RVI
    n_est = zeros(2,N);
    rho = zeros(1,N);

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
    Rbar = sum(rho);

    % Find the range of lambda
    while Rbar >= 1
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
        Rbar = sum(rho);
    end

    % Binary search
    % while abs(sum(n_high - n_low)) > 1
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
        Rbar = sum(rho);
        if Rbar >= 1
           lambda_low = lambda;
        else
           lambda_high = lambda;
        end
    end

    % Lambda+
    VF_plus = zeros(smax+1,2,N);
    n_plus = zeros(2,N);
    for j = 1:N
        p = p_pool(j);
        pe1 = pe1_pool(j);
        pe0 = pe0_pool(j);
        gamma = gamma_pool(j);
        [n_plus(:,j),VF_plus(:,:,j)] = RVI(lambda_high,p,gamma,pe1,pe0,smax,epsilon,j,w);
    end

    % Lambda-
    VF_minus = zeros(smax+1,2,N);
    n_minus = zeros(2,N);
    for j = 1:N
        p = p_pool(j);
        pe1 = pe1_pool(j);
        pe0 = pe0_pool(j);
        gamma = gamma_pool(j);
        [n_minus(:,j),VF_minus(:,:,j)] = RVI(lambda_low,p,gamma,pe1,pe0,smax,epsilon,j,w);
    end

    % Benchmark Performance
    if isequal(n_plus,n_minus)
        C_BP = 0;
        for j = 1:N
            p = p_pool(j);
            c1 = c1_pool(j);
            c2 = c2_pool(j);
            gamma = gamma_pool(j);
            [tmp,~] = expected_value(p,c1,c2,n_plus(:,j),gamma,smax,j,w);
            C_BP = C_BP + tmp;
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
            [tmp_a1,tmp_r1] = expected_value(p,c1,c2,n_plus(:,j),gamma,smax,j,w);
            [tmp_a2,tmp_r2] = expected_value(p,c1,c2,n_minus(:,j),gamma,smax,j,w);
            Pbar_plus = Pbar_plus + tmp_r1;
            Pbar_minus = Pbar_minus + tmp_r2;
            Cbar_plus = Cbar_plus + tmp_a1;
            Cbar_minus = Cbar_minus + tmp_a2;
        end
        mu = (1 - Pbar_plus) / (Pbar_minus - Pbar_plus);
        C_BP = mu * Cbar_minus + (1 - mu) * Cbar_plus;
    end

%     fprintf('------------------------------------\n');
%     fprintf('N = %d \n', N);
%     fprintf('(BP) Sum AoII = %.4f \n', Cbar);
%     fprintf('(BP) Average AoII = %.4f \n', Cbar/N);
%     fprintf('------------------------------------\n');

    % Index Ix
    % Notes: Include s=0
    Ix = zeros(smax/2,2,N);
    for j = 1:N
        p = p_pool(j);
        pe1 = pe1_pool(j);
        pe0 = pe0_pool(j);
        gamma = gamma_pool(j);
        for i = 1:smax/2
            for r = 1:2
                Trans1 = RVI_Trans(i,r,p,gamma,pe1,pe0,smax,VF_plus(:,:,j),1); % idle
                Trans2 = RVI_Trans(i,r,p,gamma,pe1,pe0,smax,VF_plus(:,:,j),2); % transmit
                Ix(i,r,j) = Trans1-Trans2-lambda_high;
            end
        end
    end

    % Policy Evaluation - IPP
    run = 15;
    epoch = 15000;
    expected_penalty = 0;
    for i = 1:run
        cumulative_penalty = 0;
        cur_age = zeros(1,N);
%         C = binornd(1,gamma_pool(1),[1,N]); % CSI % Gamma is set to be the same.
        C = realization(gamma_pool,N);
        for j = 1:epoch
            % Policy
            a = IPP(Ix,cur_age,C,N);
            % Evolve
            nxt_ages = NextStep(a,cur_age,pe1_pool,pe0_pool,C,p_pool,N); % Evolve
%             C = binornd(1,gamma_pool(1),[1,N]); % CSI % Gamma is set to be the same.
            C = realization(gamma_pool,N);
            cumulative_penalty = cumulative_penalty + f_all(cur_age,N,w);
            % one step forward
            cur_age = nxt_ages;
        end
        expected_penalty = expected_penalty + cumulative_penalty / epoch;
    end
    C_IPP = expected_penalty / run;


    run = 15;
    epoch = 15000;
    expected_penalty = 0;
    for i = 1:run
        cumulative_penalty = 0;
        cur_age = zeros(1,N);
%         C = binornd(1,gamma_pool(1),[1,N]); % CSI % Gamma is set to be the same.
        C = realization(gamma_pool,N);
        for j = 1:epoch
            % Policy
            a = Greedy_plus(cur_age,N,C);
            % Evolve
            nxt_ages = NextStep(a,cur_age,pe1_pool,pe0_pool,C,p_pool,N); % Evolve
%             C = binornd(1,gamma_pool(1),[1,N]); % CSI % Gamma is set to be the same.
            C = realization(gamma_pool,N);
            cumulative_penalty = cumulative_penalty + f_all(cur_age,N,w);
            % one step forward
            cur_age = nxt_ages;
        end
        expected_penalty = expected_penalty + cumulative_penalty / epoch;
    end
    C_GP = expected_penalty / run;
end

%% realization funtion
function ret = realization(gamma_pool,N)
    ret = zeros(1,N);
    for i = 1:N
        ret(i) = binornd(1,gamma_pool(i));
    end
end

%% Function for expected performance
% Calculate the results in Proposition 2
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

% Calculate the results in Corollary 5
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

%% Time penalty function for a single user
% All users combined
function ret = f_all(s,N,w)
    ret = 0;
    for i = 1:N
       ret = ret + f(s(i),i,w);
    end
end

% Single user
function ret = f(s,i,w)
    ret = s^w(i);
end

%% Relative Value Iteration
function [ret_policy,ret_VF] = RVI(lambda,p,gamma,pe1,pe0,smax,epsilon,j,w)
    RV_old = zeros(smax+1,2);
    action_R = zeros(smax+1,2);

    % Instant penalty with/without transmission
    C = zeros(smax+1,2);
    for i = 1:smax+1
        for a = 1:2
           C(i,a) = f(i-1,j,w) + (a-1) * lambda;
        end
    end

    cnt = 0;
    iter_max = 10000;
    H = zeros(smax+1,2);

    % Main Loop
    while cnt <= iter_max
        % Update value function
        for i = 1:smax+1
            for j = 1:2
                Trans1 = RVI_Trans(i,j,p,gamma,pe1,pe0,smax,RV_old,1); % idle
                Trans2 = RVI_Trans(i,j,p,gamma,pe1,pe0,smax,RV_old,2); % transmit
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

%% RVI_Trans in RVI
function ret = RVI_Trans(i,j,p,gamma,pe1,pe0,smax,RV_old,action)
    if i == 1 % s = 0
        ret = (1-gamma)*(1-p)*RV_old(1,1) + (1-gamma)*p*RV_old(2,1) + gamma*(1-p)*RV_old(1,2) + gamma*p*RV_old(2,2);
    elseif i ~= 1 && j == 1 % negative CSI
        if action == 2 % Transmit
            ret = (1-gamma)*(pe0*(1-p)+(1-pe0)*p)*RV_old(1,1) + (1-gamma)*(pe0*p+(1-pe0)*(1-p))*RV_old(min(i+1,smax+1),1) + gamma*(pe0*(1-p)+(1-pe0)*p)*RV_old(1,2) + gamma*(pe0*p+(1-pe0)*(1-p))*RV_old(min(i+1,smax+1),2);
        else
            ret = (1-gamma)*p*RV_old(1,1) + (1-gamma)*(1-p)*RV_old(min(i+1,smax+1),1) + gamma*p*RV_old(1,2) + gamma*(1-p)*RV_old(min(i+1,smax+1),2);
        end
    else % positive CSI
        if action == 2
            ret = (1-gamma)*(pe1*p+(1-pe1)*(1-p))*RV_old(1,1) + (1-gamma)*(pe1*(1-p)+(1-pe1)*p)*RV_old(min(i+1,smax+1),1) + gamma*(pe1*p+(1-pe1)*(1-p))*RV_old(1,2) + gamma*(pe1*(1-p)+(1-pe1)*p)*RV_old(min(i+1,smax+1),2);
        else
            ret = (1-gamma)*p*RV_old(1,1) + (1-gamma)*(1-p)*RV_old(min(i+1,smax+1),1) + gamma*p*RV_old(1,2) + gamma*(1-p)*RV_old(min(i+1,smax+1),2);
        end
    end
end

%% NextStep in RVI_Trans
function ret = NextStep(a,cur_age,pe1,pe0,C,p_pool,N)
    ret = zeros(1,N);
    for i = 1:N
        ret(i) = Evolve(a(i),cur_age(i),pe0(i),pe1(i),C(i),p_pool(i));
%         ret(i) = Evolve(a(i),cur_age(i),0,pe1(i),C(i),p_pool(i));
    end
end

%% Evolve in NextStep
% Notes: Evolution for a single user
function ret = Evolve(a,cur_age,pe0,pe1,u,p)
    if cur_age == 0
        r = binornd(1,p);
        if r == 1
            ret = 1;
        else
            ret = 0;
        end
    else
        r = binornd(1,p);
        e0 = binornd(1,pe0);
        e1 = binornd(1,pe1);
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

%% Action from Whittle's Index policy
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

%% Action from Indexed Priority policy
function a = IPP(Ix,cur_age,C,N)
    a = zeros(1,N);
    tmp = zeros(1,N);
    for i = 1:N
        tmp(i) = Ix(cur_age(i)+1,C(i)+1,i);
    end
    [~,idx] = max(tmp);
    a(idx) = 1;
end

%% Action from Greedy+ policy
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
