%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file generates the random system setups and calculates/evaluates
% the following scheduling policies.
% 1. Whittle's Index policy   (WIP)
% 2. Indexed Priority policy  (IPP)
% 3. Benchmark Performance    (BP)
% Notes: Gamma are set to be the same across all users.
% Author: Yutao Chen
% Updated: 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;

%% System Parameters
N = 5;        % Number of users in the system
N_sys = 300;  % Number of system setups
sz = [1 N];   % Size

rng(0,'twister');  % Seed

res = zeros(N_sys,3);  % Store the results

%% The case where Whittle's index policy is not feasible
for i = 1:N_sys
   fprintf('----------------------------------------------------\n');
   fprintf('System # %d \n', i);
   w = rand(sz) + 0.5;
   fprintf('w = ');
   fprintf('%g ', w);
   fprintf('\n');
   p_pool = 0.4.*rand(sz) + 0.05;
   fprintf('p_pool = ');
   fprintf('%g ', p_pool);
   fprintf('\n');
   pe0_pool = 0.45.*rand(sz);
   fprintf('pe0_pool = ');
   fprintf('%g ', pe0_pool);
   fprintf('\n');
   pe1_pool = 0.45.*rand(sz);
   fprintf('pe1_pool = ');
   fprintf('%g ', pe1_pool);
   fprintf('\n');
   gamma_pool = rand(sz);
   fprintf('gamma_pool = ');
   fprintf('%g ', gamma_pool);
   fprintf('\n');
   [C_IPP,C_BP,C_GP] = run(N,w,p_pool,pe0_pool,pe1_pool,gamma_pool);
   fprintf('(GP) Average AoII = %.4f \n', C_GP/N);
   fprintf('(IPP) Average AoII = %.4f \n', C_IPP/N);
   fprintf('(BP) Average AoII = %.4f \n', C_BP/N);
   fprintf('----------------------------------------------------\n');
   res(i,:) = [C_IPP/N C_BP/N C_GP/N];
end

%% Save the results
IPP_AVE = res(:,1);
BP_AVE = res(:,2);
GP_AVE = res(:,3);

pmin = 0.05;
pmax = 0.45;
pe1min = 0;
pe1max = 0.45;
pe0min = 0;
pe0max = 0.45;
gammamin = 0;
gammamax = 1;
wmin = 0.5;
wmax = 1.5;

filename = 'Random.mat';
save(filename,'N','N_sys','IPP_AVE','BP_AVE','GP_AVE','pmin','pmax', ...
              'pe1min','pe1max','pe0min','pe0max','gammamin', ...
              'gammamax','wmin','wmax');

%% The case when the Whittle's Index policy is feasible
pe0_pool = zeros(1,N);

for i = 1:N_sys
   fprintf('----------------------------------------------------\n');
   fprintf('System # %d \n', i);
   w = rand(sz) + 0.5;
   fprintf('w = ');
   fprintf('%g ', w);
   fprintf('\n');
   p_pool = 0.4.*rand(sz) + 0.05;
   fprintf('p_pool = ');
   fprintf('%g ', p_pool);
   fprintf('\n');
   pe1_pool = 0.45.*rand(sz);
   fprintf('pe1_pool = ');
   fprintf('%g ', pe1_pool);
   fprintf('\n');
   gamma_pool = rand(sz);
   fprintf('gamma_pool = ');
   fprintf('%g ', gamma_pool);
   fprintf('\n');
   [C_WIP,C_IPP,C_BP,C_GP] = run_Whittle(N,w,p_pool,pe0_pool,pe1_pool, ...
                                         gamma_pool);
   fprintf('(GP) Average AoII = %.4f \n', C_GP/N);
   fprintf('(WIP) Average AoII = %.4f \n', C_WIP/N);
   fprintf('(IPP) Average AoII = %.4f \n', C_IPP/N);
   fprintf('(BP) Average AoII = %.4f \n', C_BP/N);
   fprintf('----------------------------------------------------\n');
   res(i,:) = [C_GP/N C_WIP/N C_IPP/N C_BP/N];
end

%% Save the results
GP_AVE = res(:,1);
WIP_AVE = res(:,2);
IPP_AVE = res(:,3);
BP_AVE = res(:,4);

pmin = 0.05;
pmax = 0.45;
pe1min = 0;
pe1max = 0.45;
pe0 = 0;
gammamin = 0;
gammamax = 1;
wmin = 0.5;
wmax = 1.5;

filename = 'Random_Whittle.mat';
save(filename,'N','N_sys','WIP_AVE','IPP_AVE','BP_AVE','GP_AVE','pmin', ...
              'pmax','pe1min','pe1max','pe0','gammamin','gammamax', ...
              'wmin','wmax');
