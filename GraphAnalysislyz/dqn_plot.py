import numpy as np

dqn = np.loadtxt('dqn_reward.csv',delimiter = ',')

max_r = 0
i = 0
n_15 = 0
n_20 = 0
dqn_m = np.zeros(6000)
for item in dqn:
	if item > max_r:
		print 'yes',item,i
		max_r = item
	if item > 15:
		n_15 = n_15+1
	if item > 20:
		n_20 = n_20+1
	dqn_m[i] = max_r
	i = i +1 

np.savetxt('DQN_m_reward.csv',dqn_m,delimiter=',')
print 'n_15', n_15
print 'n_20', n_20
