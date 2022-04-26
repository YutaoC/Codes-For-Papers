%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize the data
% (The codes are written according to my own needs and are not universal)
% Yutao Chen
% 04/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
close all;
clearvars;

%% Import Data Here
% Example
Y1 = rand(1,50);
Y2 = rand(1,50) + 1;
Y3 = rand(1,50) + 2;
X = (1:50);

%% Plot (Single Line)
% Initialization steps:
workspace;
format long g;
format compact;
h = figure;

% Plot
plot(X, Y1, 'color',[0, 0, 0.6], 'LineWidth', 1.5,'MarkerSize',6, ...
    'MarkerFaceColor',[0, 0, 0.6]);
grid on;

% Make labels for the two axes.
xlabel('xlabel');
ylabel('ylabel');

% % Axis Range and step size
% ylim([0,12]);
% yticks(0:2:12);
% xlim([0,12]);
% xticks(0:2:12);

% Axes font and size
ax = gca;
ax.FontSize = 10;
ax.FontWeight = 'bold';

% Export as .pdf (Not perfect - Still too large margin)
filename = 'Example2';
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize', ...
    [pos(3), pos(4)])
print(h,filename,'-dpdf','-r0')

%% Plot (Multiple Lines)
% Initialization steps:
workspace;
format long g;
format compact;
h = figure;

% % Plot With Maker
% plot(X, Y1, '-o','color',[0, 0, 0.6], 'LineWidth', 1.5,'MarkerSize',6, ...
%     'MarkerFaceColor',[0, 0, 0.6]);
% hold on;
% plot(X, Y2, '-s','color',[0, 0.6, 0], 'LineWidth', 1.5,'MarkerSize',6, ...
%     'MarkerFaceColor',[0, 0.6, 0]);
% hold on;
% plot(X, Y3, '-^','color',[0.6, 0, 0], 'LineWidth', 1.5,'MarkerSize',6, ...
%     'MarkerFaceColor',[0.6, 0, 0]);
% grid on;

% Plot Without marker
plot(X, Y1, '-o','color',[0, 0, 0.6], 'LineWidth', 1.5,'Marker', 'none');
hold on;
plot(X, Y2, '-s','color',[0, 0.6, 0], 'LineWidth', 1.5,'Marker', 'none');
hold on;
plot(X, Y3, '-^','color',[0.6, 0, 0], 'LineWidth', 1.5,'Marker', 'none');
grid on;

% Make labels for the two axes.
xlabel('xlabel');
ylabel('ylabel');

% Legend
legend('1','2','3','Location','best')

% % Axis Range and step size
% ylim([0,12]);
% yticks(0:2:12);
% xlim([0,12]);
% xticks(0:2:12);

% Axes font and size
ax = gca;
ax.FontSize = 10;
ax.FontWeight = 'bold';
hold off

% Export as .pdf (Not perfect - Still too large margin)
filename = 'Example2';
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches', ...
    'PaperSize',[pos(3), pos(4)])
print(h,filename,'-dpdf','-r0')
