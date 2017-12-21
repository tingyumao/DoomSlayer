DRQN_reward = (csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DRQN_reward.csv'))';
DRQNA_reward = (csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DRQNwithA_reward.csv'))';
drqn_m_reward = csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DRQN_m_reward.csv');
drqna_m_reward = csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DRQNwithA_m_reward.csv');
dqn_reward = csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/dqn_reward.csv');
DRQN_epi = 1:6000;
DRQNA_epi = 1:6000;



% y1_add = zeros(5999,6000);
% y1 = [DRQN_reward;y1_add];
% y1_filter = ordfilt2(y1,400,ones(20,20));
% y1_dis = y1_filter(1,:);
% 
% 
% y2_add = zeros(5999,6000);
% y2 = [DRQNA_reward;y2_add];
% y2_filter = ordfilt2(y2,400,ones(20,20));
% y2_dis = y2_filter(1,:);

f1 = figure;
f2 = figure;
figure(f1);
plot(DRQN_epi, DRQN_reward,'r');
hold on;
plot(DRQN_epi,drqn_m_reward ,'y','LineWidth',8,'Color',[0.42 0.37 0.66]);
hold on;
plot(DRQNA_epi, DRQNA_reward,'b');
plot(DRQNA_epi,drqna_m_reward ,'c','LineWidth',8, 'Color',[0.97 0.43 0.18]);
h1_1 = legend('DRQN','DRQN max reward','DRQN with augmented feature','DRQN with augmented feature max reward');  
set(h1_1,'Fontsize',12);
title('Reward during training','FontSize',12);
xlabel('episode','FontSize',12)
ylabel('Reward','FontSize',12)

figure(f2);
plot(DRQN_epi, dqn_reward,'b');
hold on;
plot(DRQN_epi, DRQN_reward,'r');
hold on;
h1_1 = legend('DQN','DRQN');  
set(h1_1,'Fontsize',12);
title('Reward during training','FontSize',12);
xlabel('episode','FontSize',12)
ylabel('Reward','FontSize',12)












