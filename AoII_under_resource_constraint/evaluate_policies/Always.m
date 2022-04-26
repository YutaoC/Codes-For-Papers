%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file calculates the expected AoII of the system resulting from the
% adoption of "Always update policy" under various system parameters.
% (Both theoretical & simulation results)
% (Not given in the paper)
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% System parameters
N = 4;     % Number of states in the Markovian source
P = 0.2;   % Probability of changing state
Ps = 0.7;  % Probability of successful transmission

epoch = 100000;  % Number of epoches to be simulated

%% Expected AoII (theoretical and simulation)
AoII_theo = Theoretical(N,P,Ps);
AoII_sim = Simulation(N,P,Ps,epoch);

%% Verification (Print results)
fprintf('\n');
fprintf('AoII(Simulation) = %.4f \n', AoII_sim);
fprintf('AoII(Theoretical) = %.4f \n', AoII_theo);
fprintf('Difference = %.4f \n', abs(AoII_sim - AoII_theo));
fprintf('Percentage = %.4f%% \n', 100 * abs(AoII_sim - AoII_theo) / ...
                             mean([AoII_sim,AoII_theo]));

%% The effect of N
N = [5,7,9,11,13,15];

P = 0.2;  % Probability of changing state
Ps = 0.8; % Probability of successful transmission

epoch = 100000;  % Number of epoches to be simulated

% Main loop
AoII = zeros(2,length(N));
for i = 1:length(N)
    AoII(1,i) = Theoretical(N(i),P,Ps);
    AoII(2,i) = Simulation(N(i),P,Ps,epoch);
end

% Save the results
filename = 'AlwaysUpdatePolicy_N.mat';
save(filename, 'AoII','N','P','Ps');

%% The effect of P
P = [0,0.05,0.1,0.15,0.2,0.25,0.3,1/3];

N = 7;     % number of states in Markovian source
Ps = 0.8;  % Probability of successful transmission

epoch = 100000;  % Number of epoches to be simulated

% Main loop
AoII = zeros(2,length(P));
for i = 1:length(P)
    AoII(1,i) = Theoretical(N,P(i),Ps);
    AoII(2,i) = Simulation(N,P(i),Ps,epoch);
end

% Save the results
filename = 'AlwaysUpdatePolicy_P.mat';
save(filename, 'AoII','N','P','Ps');

%% The effect of Ps
Ps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];

N = 7;    % number of states
P = 0.2;  % Probability of changing state

epoch = 100000;  % Number of epoches to be simulated

% Main loop
AoII = zeros(2,length(Ps));
for i = 1:length(Ps)
    AoII(1,i) = Theoretical(N,P,Ps(i));
    AoII(2,i) = Simulation(N,P,Ps(i),epoch);
end

% Save the results
filename = 'AlwaysUpdatePolicy_Ps.mat';
save(filename, 'AoII','N','P','Ps');

%% Function - Theoretical
% Calculate the theoretical value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   N - Number of states in Markovian source
%   P - Probability of changing value
%   Ps - Probability of successful transmission
% Output:
%   AoII - expected Age of Incorrect Information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function AoII = Theoretical(N,P,Ps)
    Pf = 1 - Ps;  % Probability of failed transmission

    % The system of linear equations for stationary distribution
    A = zeros(N+2, N+1);  % Empty matrix A
    b = zeros(N+2,1);     % Empty vector b

    % d = 0
    A(1,1) = -2 * P;
    A(1,3) = Pf * P + Ps * (1 - 2 * P);
    for d = 2:N-1
        A(1,d+2) = Ps * (1 - 2 * P);
    end

    % d = 1
    A(2,1) = 2 * P;
    A(2,2) = -1;
    for d = 1:N-1
        A(2,d+2) = 2 * Ps * P;
    end
    A(3,2) = 1;
    A(3,3) = Pf * (1 - 2 * P) - 1;
    A(3,4) = Pf * P;
    
    % d = 2 - d = N - 3
    for d = 2:N-3
       A(d+2,d+1) = Pf * P;
       A(d+2,d+2) = Pf * (1 - 2 * P) - 1;
       A(d+2,d+3) = Pf * P;
    end

    % d = N - 2
    A(N,N-1) = Pf * P;
    A(N,N) = Pf * (1 - 2 * P) - 1;
    A(N,N+1) = 2 * Pf * P;

    % d = N - 1
    A(N+1,N) = Pf * P;
    A(N+1,N+1) = Pf * (1 - 2 * P) - 1;

    % Last Row (probabilities add up to 1)
    A(N+2,:) = ones(1,N+1);
    A(N+2,2) = 0;

    % vector b
    b(end) = 1;

    % Solve for the stationary distribution
    dist = linsolve(A(2:end,:),b(2:end));

    % For the expected AoII
    E = A(3:end-1,3:end);
    f = zeros(N-1,1);
    for d = 1:N-1
       f(d) = -d * dist(d+2); 
    end
    AoII = sum(linsolve(E,f));
end

%% Function - Simulation
% Run the simulations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%     N - number of states
%     P - probability of changing value
%     Ps - probability of successful transmission
%     epoch - total number of epoches to be simulated
% Output:
%     AoII - expected Age of Incorrect Information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function AoII = Simulation(N,P,Ps,epoch)
    % System status
    diff = 0;
    penalty = 0;
    cumulative_AoII = 0;

    % Main loop
    for i = 1:epoch
        channel_realization = randsrc(1,1,[0,1;1-Ps,Ps]);
        [diff_,penalty_] = trans(diff,penalty,channel_realization, ...
                                 N,P);
        cumulative_AoII = cumulative_AoII + penalty;
        % one step forward
        diff = diff_;
        penalty = penalty_;
    end
    AoII = cumulative_AoII / epoch;
end

%% Function - trans
% Return the next syatem state for the given current state
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%   old_diff - current difference
%   old_penalty - current penalty
%   channel_realization - 1 if transmission succeeds and 0 otherwise
%   N - number of states in the Markovian source
%   P - probability of changing value
% Output:
%   new_difference - difference at next time slot
%   new_penalty - penalty at next time slot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [new_diff, new_penalty] = trans(old_diff,old_penalty, ...
                                         channel_realization,N,P)
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
    % When the transmission succeeds
    else
        new_diff = randsrc(1,1,[0,1;1-2*P,2*P]);
        if new_diff == 0
            new_penalty = 0;
        else
            new_penalty = 1;
        end
    end
end
