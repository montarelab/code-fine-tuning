import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# read data from result csv files
mAP_5 = pd.read_csv('/home/sieberl/SA2020/pyslowfast/experiments/ex_10_500_100_v28/run-tensorboard-tag-Val_mAP.csv')
mAP_5 = mAP_5.values
mAP_5_fast = pd.read_csv('/home/sieberl/SA2020/pyslowfast/experiments/ex_10_500_100_v30/run-tensorboard-tag-Val_mAP'
                         '.csv')
mAP_5_fast = mAP_5_fast.values
mAP_5_4 = pd.read_csv('/home/sieberl/SA2020/pyslowfast/experiments/ex_10_500_100_v29/run-tensorboard-tag-Val_mAP'
                         '.csv')
mAP_5_4 = mAP_5_4.values


#mAP = mAP[:,2]



"""
loss = pd.read_csv('/home/sieberl/SA2020/pyslowfast/experiments/ex_10_500_100_v16/run-tensorboard-tag-Train_loss.csv')
loss = loss.values
#loss = loss[:,2]

# more sophisticated loss sampling method
first = loss[0,1]
last = loss[-1,1]
loss[:,1] = loss[:,1] - first
loss[:,1] = loss[:,1] / (last - first) * 14
loss_final = loss[:,2]
batch = loss[:,1]
# find the loss array with an average value per epoch

"""
mAP_5 = mAP_5[:,2] * 100
mAP_5_fast = mAP_5_fast[:,2] * 100
mAP_5_4 = mAP_5_4[:,2] * 100

mAP_5_p = np.array([76.26, 86.41, 95.53, 96.46, 96.65])
mAP_5_fast_p = np.array([76.89, 93.75, 95.77, 96.27, 96.36])
mAP_5_4_p = np.array([63.28, 75.02, 80.00, 83.43, 86.48])

# Create some mock data
epoch = np.arange(1, 9, 1)
epoch_p = np.array([1,2,3,4,8])
fig, ax1 = plt.subplots(figsize=(7,3.5))

ax1.set_xlabel('epoch')
ax1.set_ylabel('mAP(%)')
ax1.set_title('Stepwise layer opening')
ax1.grid()
lns1 = ax1.plot(epoch, mAP_5, color='#4267B2', label='Figure 2, c) - Val')
lns2 = ax1.plot(epoch, mAP_5_fast, color='grey', label='Figure 2, d) - Val')
lns3 = ax1.plot(epoch, mAP_5_4, color='lightskyblue', label='Figure 2, e) - Val')
lns1_p = ax1.plot(epoch_p, mAP_5_p, color='#4267B2', marker='x', linestyle='dashed', label='Figure 2, c) - Train')
#lns1_p = ax1.scatter(epoch_p, mAP_5_p, color='#4267B2', marker='o', linestyle='dashed', label='Figure 2, c) - Train')
#maybe take scatter instead of plot

lns2_p = ax1.plot(epoch_p, mAP_5_fast_p, color='grey', marker='x', linestyle='dashed', label='Figure 2, d) - Train')

lns3_p = ax1.plot(epoch_p, mAP_5_4_p, color='lightskyblue', marker='x', linestyle='dashed', label='Figure 2, '
                                                                                                  'e) - Train')
ax1.tick_params(axis='y')


"""
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('mAP(%)')  # we already handled the x-label with ax1
plt.ylim(ymax=47, ymin=37)
lns2 = ax2.plot(epoch, mAP_final, marker='o', color='#4267B2', label='mAP (Validation)')
ax2.tick_params(axis='y')
ax2.grid()


lns = lns1 + lns2
labs = [l.get_label() for l in lns]
l = ax2.legend(lns, labs, loc=0)
l.set_zorder(100)
"""
#ax1.legend()
ax1.legend(loc='upper center', bbox_to_anchor=(0.5,-0.13), fancybox=False, frameon=False, shadow=True, ncol=2)

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('/home/sieberl/SA2020/pyslowfast/report_visualizations/layers.jpg', bbox_inches='tight')