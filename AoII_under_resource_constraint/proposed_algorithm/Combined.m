%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file implements the algorithm for finding the optimal policy for
% the constrained optimization problem.
% See Section III.E of the paper
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% System parameters
N = 7;              % number of states
P = 0.2;            % Probability of changing state
Ps = 0.8;           % Probability of successful transmission
alpha_pool = 0.02;  % Power budget

% Precision parameters
smax = 800;      % Upper bound of the penalty (Truncation precision)
epsilon = 0.01;  % Tolerance (RVI precision)
xi = 0.01;       % Small pertubation of lambda (Lambda search precision)

% Store the results
pool_size = length(alpha_pool);
lambda_high_pool = zeros(1,pool_size);  % Lambda+
AoII_high_pool = zeros(1,pool_size);    % C+
Ratio_high_pool = zeros(1,pool_size);   % R+
lambda_low_pool = zeros(1,pool_size);   % Lambda-
AoII_low_pool = zeros(1,pool_size);     % C-
Ratio_low_pool = zeros(1,pool_size);    % R-
mu_pool = zeros(1,pool_size);           % mu
AoII_pool = zeros(1,pool_size);         % Minimum AoII

%% Auxiliary quantities calculations
% Minimum penalty constraint for each difference
bound = zeros(1,N);
for d = 2:N
   bound(d) = bound(d-1) + d - 1; 
end

% Transition Probability Matrix for RVI
[P_trans,P_notrans] = trans_matrix(N,P,Ps,bound,smax);

%% Main Loop
for i = 1:pool_size
    alpha = alpha_pool(i);
    fprintf('Current Power Budget: %.4f\n', alpha);
    % Initalization
    lambda_low = 0;
    lambda_high = 1;
    n_est = RVI(N, bound, lambda_low, smax, epsilon,P_trans,P_notrans);
    [~,Pbar] = Evaluate(N,P,Ps,n_est,bound,0);

    % Find the range of lambda
    fprintf('...Finding the range of lambda\n');
    while Pbar >= alpha
        lambda_low = lambda_high;
        lambda_high = lambda_low * 2;
        n_est = RVI(N, bound, lambda_high, smax, epsilon,P_trans,P_notrans);
        [~,Pbar] = Evaluate(N,P,Ps,n_est,bound,0);
    end

    % Binary search
    fprintf('...Binary searching\n');
    while lambda_high - lambda_low >= xi
        lambda = (lambda_low + lambda_high) / 2;
        n_est = RVI(N, bound, lambda, smax, epsilon,P_trans,P_notrans);
        [~,Pbar] = Evaluate(N,P,Ps,n_est,bound,0);
        if Pbar >= alpha
           lambda_low = lambda; 
        else
           lambda_high = lambda;
        end
    end

    % Lambda + Pertubation
    n_plus = RVI(N, bound, lambda_high, smax, epsilon,P_trans,P_notrans);

    % Lambda - Pertubation
    n_minus = RVI(N, bound, lambda_low, smax, epsilon,P_trans,P_notrans);

    if isequal(n_plus,n_minus)
        % Evaluate the performance
        [Cbar,Pbar] = Evaluate(N,P,Ps,n_plus,bound,1);
        AoII_high_pool(i) = Cbar;
        AoII_low_pool(i) = Cbar;
        Ratio_high_pool(i) = Pbar;
        Ratio_low_pool(i) = Pbar;

        % Print the results
        fprintf('Lambda + Pertubation: %.4f\n', lambda_high);
        lambda_high_pool(i) = lambda_high;
        fprintf('Lambda - Pertubation: %.4f\n', lambda_low);
        lambda_low_pool(i) = lambda_low;
        
        fprintf('No randomization needed.\n');
        mu_pool(i) = 1;
        fprintf('Threshold Vector: %s\n', sprintf('%d ', n_plus));
        
        fprintf('Expected AoII: %.4f\n', Cbar);
        AoII_pool(i) = Cbar;
        fprintf('Expected Transmission Ratio: %.4f\n', Pbar);
        fprintf('\n');
    else
        % Evaluate the performance
        [Cbar_plus,Pbar_plus] = Evaluate(N,P,Ps,n_plus,bound,1);
        [Cbar_minus,Pbar_minus] = Evaluate(N,P,Ps,n_minus,bound,1);

        % Print the results
        fprintf('Lambda + Pertubation: %.4f\n', lambda_high);
        lambda_high_pool(i) = lambda_high;
        fprintf('Threshold Vector: %s\n', sprintf('%d ', n_plus));
        fprintf('Expected AoII +: %.4f\n', Cbar_plus);
        AoII_high_pool(i) = Cbar_plus;
        fprintf('Expected Transmission Ratio +: %.4f\n', Pbar_plus);
        Ratio_high_pool(i) = Pbar_plus;
        
        fprintf('Lambda - Pertubation: %.4f\n', lambda_low);
        lambda_low_pool(i) = lambda_low;
        fprintf('Threshold Vector: %s\n', sprintf('%d ', n_minus));
        fprintf('Expected AoII -: %.4f\n', Cbar_minus);
        AoII_low_pool(i) = Cbar_minus;
        fprintf('Expected Transmission Ratio -: %.4f\n', Pbar_minus);
        Ratio_low_pool(i) = Pbar_minus;

        % Obtain the mixing coefficient and the final results
        mu = (alpha - Pbar_plus) / (Pbar_minus - Pbar_plus);
        Cbar = mu * Cbar_minus + (1 - mu) * Cbar_plus;
        fprintf('mu: %.4f\n', mu);
        mu_pool(i) = mu;
        fprintf('Expected AoII: %.4f\n', Cbar);
        AoII_pool(i) = Cbar;
        fprintf('Expected Transmission Ratio: %.4f\n', alpha);
        fprintf('\n');
    end
end

%% Save the results
filename = 'OptimalPolicies.mat';
save(filename, 'lambda_high_pool','AoII_high_pool','Ratio_high_pool', ...
               'lambda_low_pool','AoII_low_pool','Ratio_low_pool', ...
               'mu_pool','AoII_pool','alpha_pool','N','P','Ps','smax', ...
               'epsilon',"xi");
