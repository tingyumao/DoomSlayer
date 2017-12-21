DQNP_reward = (csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DQNP_reward_plot_6000.csv'))';
DDQNP_reward = (csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DDQNP_reward_plot.csv'))';
DQNP_loss = (csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DQNP_loss_plot_6000.csv'))';
DDQNP_loss = (csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/DDQNP_loss_plot.csv'))';
ddqn_reward = load('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/ddqn_reward.csv');
dqn_reward = csvread('/Users/lingyuzhang/Spring17/LuckyForFinal/1ReinforementLearning/CourseProject/plot/dqn_reward.csv');
DQNP_epi = 1:6000;
DDQNP_epi = 1:6000;

DQNP_time_step = 1:482349;
DDQNP_time_step = 1:458175;

f1 = figure;
f2 = figure;
f3 = figure;
f4 = figure;
% figure(f1);
% plot(DQNP_epi, y1_dis, 'c');
% hold on;
% plot(DDQNP_epi, y2_dis,'b');
figure(f1);
plot(DQNP_epi, smooth(DQNP_reward,10,'loess'),'r');
hold on;
plot(DDQNP_epi, smooth(DDQNP_reward,10,'loess'),'b');
h1_1 = legend('DQN + prioritized experience replay','DDQN + prioritized experience replay');  
set(h1_1,'Fontsize',12);
title('Reward during training','FontSize',12);
xlabel('episode','FontSize',12)
ylabel('Reward','FontSize',12)



figure(f2);
plot(DQNP_time_step, medfilt1(DQNP_loss,20),'r');
hold on;
plot(DDQNP_time_step, medfilt1(DDQNP_loss,20),'b');
h2_1 = legend('DQN + prioritized experience replay','DDQN + prioritized experience replay'); 
set(h2_1,'Fontsize',12);
title('Loss function during training','FontSize',12);
xlabel('time step','FontSize',12)
ylabel('Loss','FontSize',12)

figure(f3);
plot(DQNP_epi, smooth(DDQNP_reward,10,'loess'),'b');
hold on;
plot(DDQNP_epi, smooth(ddqn_reward,10,'loess'),'r');
h2_1 = legend('DDQN + prioritized experience replay','DDQN'); 
set(h2_1,'Fontsize',12);
title('Reward during training','FontSize',12);
xlabel('episode','FontSize',12)
ylabel('Reward','FontSize',12)

figure(f4);
plot(DQNP_epi, smooth(DQNP_reward,10,'loess'),'b');
hold on;
plot(DDQNP_epi, smooth(dqn_reward,10,'loess'),'r');
h2_1 = legend('DQN + prioritized experience replay','DQN'); 
set(h2_1,'Fontsize',12);
title('Reward during training','FontSize',12);
xlabel('episode','FontSize',12)
ylabel('Reward','FontSize',12)


% y1_add = zeros(1999,2000);
% y1 = [DQNP_reward;y1_add];
% y1_filter = ordfilt2(y1,33,ones(6,6));
% y1_dis = y1_filter(1,:);
% 
% y2_add = zeros(5999,6000);
% y2 = [DDQNP_reward;y2_add];
% y2_filter = ordfilt2(y2,33,ones(6,6));
% y2_dis = y2_filter(1,:);
% y3_add = zeros(155545,155546);
% y3 = [DQNP_loss;y3_add];
% 
% y4_add = zeros(458174,458175);
% y4 = [DQNP_loss;y4_add];






