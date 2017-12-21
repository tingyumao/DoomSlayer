import numpy as np

drqn = np.loadtxt('DRQN_output.txt',skiprows=60, delimiter='-',usecols=(2))

# np.savetxt('DRQN_reward.csv',drqn,delimiter=',')


drqnwithA = np.loadtxt('DRQNwithA_output.txt',skiprows=60, delimiter='-',usecols=(2))

max_r_1 = 0
j = 0
drqn_m = np.zeros(6000)
for item in drqn:
	if item > max_r_1:
		print 'yes',item,j
		max_r_1 = item
	drqn_m[j] = max_r_1
	j = j +1 

max_r = 0
i = 0
n_15 = 0
n_20 = 0
drqnwithA_m = np.zeros(6000)
for item in drqnwithA:
	if item > max_r:
		print 'yes',item,i
		max_r = item
	drqnwithA_m[i] = max_r
	i = i +1 

np.savetxt('DRQN_m_reward.csv',drqn_m,delimiter=',')
np.savetxt('DRQNwithA_m_reward.csv',drqnwithA_m,delimiter=',')
