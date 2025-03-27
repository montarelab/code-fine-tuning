import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# read data from result csv files
loss_class = pd.read_csv('/home/sieberl/SA2020/pyslowfast/experiments/da_10_5000_200_v1/run-tensorboard-tag'
                     '-Train_loss_class'
                   '.csv')
loss_class = loss_class.values

loss_rot = pd.read_csv('/home/sieberl/SA2020/pyslowfast/experiments/da_10_5000_200_v1/run-tensorboard-tag'
                     '-Train_loss_rot'
                   '.csv')
loss_rot = loss_rot.values



batch_class = loss_class[:,1]
batch_rot = loss_rot[:,1]

loss_class = loss_class[:,2]
loss_rot = loss_rot[:,2]

loss_class_avg = np.zeros((loss_class.shape[0],))
loss_rot_avg = np.zeros((loss_rot.shape[0],))

averaging = 50
for i in range(loss_class.shape[0]):
    if i < averaging:
        loss_class_avg[i] = np.mean(loss_class[:i])
        loss_rot_avg[i] = np.mean(loss_rot[:i])
    else:
        loss_class_avg[i] = np.mean(loss_class[i-averaging:i])
        loss_rot_avg[i] = np.mean(loss_rot[i-averaging:i])





#loss = loss[:,2]

"""
# more sophisticated loss sampling method
first = loss[0,1]
last = loss[-1,1]
loss[:,1] = loss[:,1] - first
loss[:,1] = loss[:,1] / (last - first) * 14
loss_final = loss[:,2]
batch = loss[:,1]
"""

# find the loss array with an average value per epoch
"""
batches_per_epoch = int(loss.shape[0] / mAP.shape[0])
loss_final = np.zeros((mAP.shape[0],))

for i in range(loss_final.shape[0]):
    print(loss[i*batches_per_epoch:(i+1)*batches_per_epoch,2].shape)
    average = np.mean(loss[i*batches_per_epoch:(i+1)*batches_per_epoch,2])
    loss_final[i] = average
"""


"""

mAP_final = mAP[:,2] * 100

# Create some mock data
epoch = np.arange(1, 15, 1)

"""


fig, ax1 = plt.subplots(figsize=(6,3.5))

ax1.set_xlabel('batch')
ax1.set_ylabel('Entropy (batch average)')
ax1.set_title('Training curves in combined setting')
lns1 = ax1.plot(batch_class, loss_class, color='lightskyblue', alpha=0.3, label='Action BCE (Training)')
lns1_b = ax1.plot(batch_class, loss_class_avg, color='lightskyblue', label='Action BCE 3k-avg (Training)')
ax1.tick_params(axis='y')
plt.ylim(ymax=1.75,ymin=0)
ax1.grid()
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

#ax2.set_ylabel('Cross Entropy (batch average)')  # we already handled the x-label with ax1
#plt.ylim(ymax=47, ymin=37)
lns2 = ax1.plot(batch_rot, loss_rot, color='#4267B2', alpha=0.3, label='Rotation CE (Training)')
lns2_b = ax1.plot(batch_rot, loss_rot_avg, color='#4267B2', label='Rotation CE 3k-avg (Training)')
#ax2.tick_params(axis='y')
#ax2.grid()


lns = lns1 + lns1_b + lns2 + lns2_b
labs = [l.get_label() for l in lns]
l = ax1.legend(loc='upper center', bbox_to_anchor=(0.5,-0.20), fancybox=False, frameon=False, shadow=True, ncol=2)
l.set_zorder(100)



fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('/home/sieberl/SA2020/pyslowfast/report_visualizations/combined_training_curves.jpg', bbox_inches='tight')
