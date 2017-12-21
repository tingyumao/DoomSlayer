ddqn_reward = csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/ddqn_reward.csv');
dqn_reward = csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/dqn_reward.csv');
ddqn_m_reward = csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DDQN_m_reward.csv');
dqn_m_reward = csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DQN_m_reward.csv');

DDQN_epi = 1:6000;
dqn_epi = 1:6000;


f1 = figure;
figure(f1);
plot(dqn_epi,dqn_reward ,'b');
hold on;
plot(dqn_epi,dqn_m_reward ,'y','LineWidth',8,'Color',[0.42 0.37 0.66]);
hold on;
plot(DDQN_epi, ddqn_reward,'r');
hold on;
plot(dqn_epi,ddqn_m_reward ,'c','LineWidth',8, 'Color',[0.97 0.43 0.18]);
h1_1 = legend('DQN','DQN max reward','DDQN','DDQN max reward');  
set(h1_1,'Fontsize',12);
title('Reward during training','FontSize',12);
xlabel('episode','FontSize',12)
ylabel('Reward','FontSize',12)
%xlim([0 500]);