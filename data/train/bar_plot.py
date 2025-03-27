"""This script creates the baseline bar plot to compare three experiments
- fb model predictions on reduced detections dataset
- fb model preidctions on reduced gt dataset
- fb model values from their paper"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['watch \n (a person)',
          'talk to (e.g., self, \n a person, a group)',
          'listen to \n (a person)',
          'sit',
          'carry/hold \n (an object)',
          'walk',
          'touch \n (an object)',
          'bend/bow \n (at the waist)',
          'lie/sleep',
          'ride',
          '',
          'mAP \n (10 classes)']

ava_detected = [16.45, 49.87, 30.40, 24.22, 34.54, 68.44, 42.91, 44.38, 66.53, 61.00, 0, 43.87]
ava_gt = [40.49, 52.71, 50.20, 35.30, 43.73, 85.77, 62.44, 81.12, 95.10, 91.63, 0, 63.90]
fb_detected = [63.00, 78.00, 60.00, 76.00, 50.00, 72.00, 30.00, 36.00, 39.00, 38.00, 0]

fb_detected_mean = sum(fb_detected) / len(fb_detected)
fb_detected.append(fb_detected_mean)


x = np.arange(len(labels))  # the label locations
width = 0.42  # the width of the bars

fig, ax = plt.subplots(figsize=(12,5))

#plt.grid(axis='y')

rects1 = ax.bar(x - 3/4 * width, ava_detected, width / 3 * 2, label='Detected boxes reduced AVA dataset',
                color='lightskyblue') ##87CEFA
rects2 = ax.bar(x, ava_gt, width / 3 * 2, label='Ground-truth boxes reduced AVA dataset', color='grey')
rects3 = ax.bar(x + 3/4 * width, fb_detected, width / 3 * 2, label='Detected boxes entire AVA dataset',
                color='#4267B2') #slategrey - #778899

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AP(%)')
plt.ylim(ymax=115, ymin=0)
ax.set_title('Per-class performance comparison')
ax.set_xlabel('Classes')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=55)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    group = 0
    for rect in rects:
        group += 1
        if group == 11:
            continue
        height = rect.get_height()
        ax.annotate('{:0.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.axvline(x=10, c='k', lw=0.5, linestyle='-')

#fig.tight_layout()


plt.savefig('/home/sieberl/SA2020/pyslowfast/report_visualizations/barplot.jpg', bbox_inches='tight')