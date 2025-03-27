import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
# read data from result csv files
mAP = pd.read_csv('/home/sieberl/SA2020/pyslowfast/experiments/ex_10_500_100_v16/run-tensorboard-tag-Val_mAP.csv')
mAP = mAP.values
#mAP = mAP[:,2]

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

batches_per_epoch = int(loss.shape[0] / mAP.shape[0])
loss_final = np.zeros((mAP.shape[0],))

for i in range(loss_final.shape[0]):
    print(loss[i*batches_per_epoch:(i+1)*batches_per_epoch,2].shape)
    average = np.mean(loss[i*batches_per_epoch:(i+1)*batches_per_epoch,2])
    loss_final[i] = average

mAP_final = mAP[:,2] * 100
"""

mAP_ava = np.array([37.70, 42.74, 48.02, 49.79, 49.26])
mAP_ava_interp = np.array([30.06, 37.70, (37.70 + 42.74)/2, 42.74, (42.74 + 48.02)/2, 48.02, (48.02 + 49.79)/2,
                           49.79, (49.79 + 49.26)/2, 49.26])
mAP_ava_combined = np.array([30.06, 33.74, 38.42, 41.29, 40.38, 40.99, 46.01, 42.39, 44.48, 44.58])

mAP_kinetics = np.array([27.22, 26.96, 28.73, 29.06, 29.38])
mAP_kinetics_interp = np.array([27.22, 27.22, (27.22 + 26.96)/2, 26.96, (26.96 + 28.73)/2, 28.73, (28.73 + 29.06)/2,
                               29.06, (29.06 + 29.38)/2, 29.38])
mAP_kinetics_combined = np.array([27.34, 27.71, 29.44, 30.11, 29.66, 30.16, 32.56, 31.82, 32.78, 32.74])

# Create some mock data
#epoch = np.arange(1, 6, 1)
epoch = np.arange(2,12,2)
epoch_combined = np.arange(1,11,1)


fig, ax1 = plt.subplots(figsize=(7,5))

ax1.set_xlabel('epoch (domain adaptation training)')
ax1.set_ylabel('mAP(%)')
plt.ylim(ymax=53, ymin=25)
ax1.set_title('Validation on both domains for both runs')
lns1 = ax1.plot(epoch, mAP_ava, marker='o', color='lightskyblue', label='AVA Validation')
lns1b = ax1.plot(epoch, mAP_kinetics, marker='x', color='lightskyblue', label='Kinetics Validation')
lns2 = ax1.plot(epoch_combined, mAP_ava_combined, marker='o', color='#4267B2', label='Domain adaptation AVA '
                                                                                     'Validation ')
lns2b = ax1.plot(epoch_combined, mAP_kinetics_combined, marker='x', color='#4267B2', label='Domain adaptation '
                                                                                           'Kinetics Validation')

# do the fill between the rooms with preparation
t = epoch_combined
ax1.fill_between(t, mAP_kinetics_combined, mAP_kinetics_interp, facecolor='palegreen', alpha=0.5)
ax1.fill_between(t, mAP_ava_interp, mAP_ava_combined, facecolor='lightcoral', alpha=0.5)





plt.xticks(epoch_combined)
ax1.tick_params(axis='y')
ax1.grid()

ax1.annotate('49.26%', xy=(5, 48),  xycoords='data',
            xytext=(1, 0.94), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            )
ax1.annotate('44.58%', xy=(10, 43),  xycoords='data',
            xytext=(1, 0.66), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            )

ax1.annotate('29.38%', xy=(5, 28),  xycoords='data',
            xytext=(1, 0.13), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            )

ax1.annotate('32.74%', xy=(10, 32),  xycoords='data',
            xytext=(1, 0.33), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top',
            )

"""
ax1.annotate('local max', xy=(5, 48),  xycoords='data',
            xytext=(0.5, 0.8), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', width=3, headwidth=5, shrink=0.001),
            horizontalalignment='right', verticalalignment='top',
            )
            
"""
"""
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_xlabel('epoch combined')  # we already handled the x-label with ax1
plt.ylim(ymax=47, ymin=20)
lns2 = ax2.plot(epoch_combined, mAP_ava_combined, marker='o', color='#4267B2', label='mAP(%) combined')
ax2.tick_params(axis='x')
ax2.grid()
"""

lns = lns1 + lns1b + lns2 + lns2b
labs = [l.get_label() for l in lns]
l = ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5,-0.1), fancybox=False, frameon=False, shadow=True, \
                                                                                                            ncol=2)
l.set_zorder(100)

ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(epoch)
ax2.set_xticklabels(np.arange(1,6,1))
ax2.set_xlabel('epoch (raw training)')

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('/home/sieberl/SA2020/pyslowfast/report_visualizations/combined_curves_2axis.jpg', bbox_inches='tight')
