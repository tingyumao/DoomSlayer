a3c_reward = load('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/a3c_line.mat');
% DRQNA_reward = (csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DRQNwithA_reward.csv'))';
DRQN_reward = (csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DRQN_reward.csv'))';
a3c = a3c_reward.reward;
a3c_epi = 1:6:5694;
DRQN_epi = 1:6000;

f1 = figure;
figure(f1);
plot(a3c_epi, smooth(a3c,20,'loess'),'r');
hold on;
plot(DRQN_epi, smooth(DRQN_reward,20,'loess'),'b');
h1_1 = legend('A3C','DRQN');  
set(h1_1,'Fontsize',12);
title('Reward during training','FontSize',12);
xlabel('episode','FontSize',12)
ylabel('Reward','FontSize',12)