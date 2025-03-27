"""Creates the annotation files for a certain configuration given the original ava annotation and frame list files"""
import os
import shutil
import numpy as np
from functions import reduce_pbtxt_write_json
import pandas as pd

import csv

orig_annot = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations'
orig_frame_lists = '/srv/beegfs02/scratch/da_action/data/kinetics700/frame_lists'

#############################
# Hyperparameters
work_dir = '/srv/beegfs02/scratch/da_action/data/kinetics700'

SAMPLES_TRAIN = 500
#SAMPLES_TRAIN = 400
SAMPLES_VAL = 500
#SAMPLES_VAL = 80

# 10 class scenario
# watch (a person) -> 80
# talk to (e.g., self, a person, a group) -> 79
# listen to (a person) -> 74
# sit -> 11
# carry/hold (an object) -> 17
# walk -> 14
# touch (an object) -> 59
# bend/bow (at the waist) -> 1
# lie/sleep -> 8
# ride (e.g., a bike, a car, a horse) -> 49

CLASSES = [80,79,74,11,17,14,59,1,8,49]

##############################
# new class ids (introduces enumeration from 1-10 given that it is for a total of 10 classes)
new_classes = [i + 1 for i in range(len(CLASSES))]

# generate config name
config_name = str(len(CLASSES)) + '_' + str(SAMPLES_TRAIN) + '_' + str(SAMPLES_VAL)

# Remove annotations directory and frame list directory if they already exist
annot_dir = work_dir + '/' + 'annotations_' + config_name
frame_list_dir = work_dir + '/' + 'frame_lists_' + config_name

# Delete directories if they already exist
if os.path.exists(annot_dir):
    shutil.rmtree(annot_dir)
if os.path.exists(frame_list_dir):
    shutil.rmtree(frame_list_dir)

# Create new annotation and frame_lists folders
os.mkdir(work_dir + '/' + 'annotations_' + config_name)
os.mkdir(work_dir + '/' + 'frame_lists_' + config_name)

# copy and reduce ava_action_list_v2.2.pbtxt file to wanted classes
reduce_pbtxt_write_json(orig_annot=orig_annot, annot_dir=annot_dir, classes=CLASSES)


# copy ava_included_timestamps_v2.2.txt
original = os.path.join(orig_annot, 'ava_included_timestamps_v2.2.txt')
target = os.path.join(annot_dir, 'ava_included_timestamps_v2.2.txt')
shutil.copyfile(original, target)

# TRAINING
# copy ava_train_excluded_timestamps_v2.2.csv
original = os.path.join(orig_annot, 'kinetics_train_excluded_timestamps_v2.1.csv')
target = os.path.join(annot_dir, 'kinetics_train_excluded_timestamps_v2.1.csv')
shutil.copyfile(original, target)


# load ava_train_v2.2.csv
# think of not loading the entire file but only a certain number of lines
#ava_train = pd.read_csv(os.path.join(orig_annot, 'ava_train_v2.2.csv'), header=None, nrows=400*100*SAMPLES_TRAIN)
ava_train = pd.read_csv(os.path.join(orig_annot, 'kinetics_train_v2.1.csv'), header=None)

# remove the excluded timestamps
#IT SEEMS THAT THE NONVALID TIMESTAMPS HAVE ALREADY BEEN EXCLUDED (OR THEY ARE ANYWAY NOT PART OF THE ANNOTATION FILES)
"""
ava_train_excl = pd.read_csv(os.path.join(orig_annot, 'kinetics_train_excluded_timestamps_v2.1.csv'), header=None)
for index, row in ava_train_excl.iterrows():
    indexNames = ava_train[(ava_train[0] == row[0]) & (ava_train[1] == row[1])].index
    ava_train.drop(indexNames, inplace=True)
"""

# extract the number of wanted samples from the file
frames = []
# iterate through list of wanted classes
for cl in CLASSES:
    frames.append(ava_train.loc[ava_train[6] == cl].head(SAMPLES_TRAIN))
ava_train = pd.concat(frames)

# sort for the first two columns
ava_train = ava_train.sort_values(by=[0,1])

# do class id replacement to have subsequent class ids starting at 1
helper = ava_train[6]

for i in range(len(CLASSES)):
    helper = helper.replace(to_replace=CLASSES[i], value=new_classes[i] + 100)

for index, value in helper.iteritems():
    helper[index] = value - 100

ava_train[6] = helper

# save to a file
ava_train.to_csv(os.path.join(annot_dir, 'kinetics_train_v2.1.csv'), header=None, index=False, float_format='%.3f')
print('after kinetics_train_v2.1.csv')

# load ava_train_predicted_boxes.csv even if it is probably not used in actual project
ava_train_pred = pd.read_csv(os.path.join(orig_annot, 'kinetics_train_predicted_boxes.csv'), header=None)
#ava_train_pred = pd.read_csv(os.path.join(orig_annot, 'ava_train_predicted_boxes.csv'), header=None,
#                             nrows=80*150*SAMPLES_TRAIN)

"""
frames = []
for index, row in ava_train.iterrows():
    a = ava_train_pred.loc[(ava_train_pred[0] == row[0]) & (ava_train_pred[1] == row[1])]
    if a.empty:
        a = pd.DataFrame(index=np.arange(1), columns=np.arange(2))
        a[0] = row[0]
        a[1] = row[1]
        print(a)
    frames.append(a)
ava_train_pred = pd.concat(frames)

# save to a file
ava_train_pred.to_csv(os.path.join(annot_dir, 'kinetics_train_predicted_boxes.csv'), header=None,
                      index=False, float_format='%.3f')
"""



#ACTIVATE THIS SECTION AGAIN
f = open(os.path.join(annot_dir, 'kinetics_train_predicted_boxes.csv'), 'w')
with f:
    writer = csv.writer(f)

    for index, row in ava_train.iterrows():
        a = ava_train_pred.loc[(ava_train_pred[0] == row[0]) & (ava_train_pred[1] == row[1])]
        if a.empty:
            a = [row[0], row[1]]
            writer.writerow(a)
        else:
            x = a.values.tolist()
            for i in x:
                for j in range(2,6):
                    i[j] = round(i[j], 3)
                i[6] = ''
                i[7] = round(i[7], 3)
                writer.writerow(i)




print('after kinetics_train_predicted_boxes.csv')












# VALIDATION
# copy ava_train_excluded_timestamps_v2.2.csv

original = os.path.join(orig_annot, 'kinetics_val_excluded_timestamps_v2.1.csv')
target = os.path.join(annot_dir, 'kinetics_val_excluded_timestamps_v2.1.csv')
shutil.copyfile(original, target)


# load ava_val_v2.2.csv
# think of not loading the entire file but only a certain number of lines
#ava_val = pd.read_csv(os.path.join(orig_annot, 'ava_val_v2.2.csv'), header=None, nrows=80*100*SAMPLES_VAL)
ava_val = pd.read_csv(os.path.join(orig_annot, 'kinetics_val_v2.1.csv'), header=None)

# remove the excluded timestamps
#IT SEEMS THAT THE NONVALID TIMESTAMPS HAVE ALREADY BEEN EXCLUDED (OR THEY ARE ANYWAY NOT PART OF THE ANNOTATION FILES)
"""
ava_val_excl = pd.read_csv(os.path.join(orig_annot, 'kinetics_val_excluded_timestamps_v2.1.csv'), header=None)
for index, row in ava_val_excl.iterrows():
    indexNames = ava_val[(ava_val[0] == row[0]) & (ava_val[1] == row[1])].index
    ava_val.drop(indexNames, inplace=True)
"""
# extract the number of wanted samples from the file
frames = []
# iterate through list of wanted classes
for cl in CLASSES:
    frames.append(ava_val.loc[ava_val[6] == cl].head(SAMPLES_VAL))
ava_val = pd.concat(frames)

# sort for the first two columns
ava_val = ava_val.sort_values(by=[0,1])

# do class id replacement to have subsequent class ids starting at 1
helper = ava_val[6]

for i in range(len(CLASSES)):
    helper = helper.replace(to_replace=CLASSES[i], value=new_classes[i] + 100)

for index, value in helper.iteritems():
    helper[index] = value - 100

ava_val[6] = helper

# save to a file
ava_val.to_csv(os.path.join(annot_dir, 'kinetics_val_v2.1.csv'), header=None, index=False, float_format='%.3f')

print('after kinetics_val_v2.1.csv')
#load ava_val_predicted_boxes.csv -> this is definitely used in the classification
#ava_val_pred = pd.read_csv(os.path.join(orig_annot, 'ava_val_predicted_boxes.csv'), header=None,
# nrows=80*150*SAMPLES_VAL)
ava_val_pred = pd.read_csv(os.path.join(orig_annot, 'kinetics_val_predicted_boxes.csv'), header=None)

"""
frames = []
for index, row in ava_val.iterrows():
    frames.append(ava_val_pred.loc[(ava_val_pred[0] == row[0]) & (ava_val_pred[1] == row[1])])
ava_val_pred = pd.concat(frames)

# save to a file
ava_val_pred.to_csv(os.path.join(annot_dir, 'kinetics_val_predicted_boxes.csv'), header=None,
                      index=False, float_format='%.3f')

"""

f = open(os.path.join(annot_dir, 'kinetics_val_predicted_boxes.csv'), 'w')
with f:
    writer = csv.writer(f)

    for index, row in ava_val.iterrows():
        a = ava_val_pred.loc[(ava_val_pred[0] == row[0]) & (ava_val_pred[1] == row[1])]
        if a.empty:
            a = [row[0], row[1]]
            writer.writerow(a)
        else:
            x = a.values.tolist()
            for i in x:
                for j in range(2,6):
                    i[j] = round(i[j], 3)
                i[6] = ''
                i[7] = round(i[7], 3)
                writer.writerow(i)
print('after kinetics_val_predicted_boxes.csv')

# adapt train frame_lists such that only the present videos are considered
ava_train_fl = pd.read_csv(os.path.join(orig_frame_lists, 'train.csv'), delimiter=' ')
unique_video_ids = ava_train[0].unique().tolist()

frames = []
for id in unique_video_ids:
    frames.append(ava_train_fl.loc[ava_train_fl['original_vido_id'] == id])

ava_train_fl = pd.concat(frames)
ava_train_fl = ava_train_fl.replace(np.nan, -1, regex=True)
ava_train_fl.to_csv(os.path.join(frame_list_dir, 'train.csv'), sep=' ', index=False)

print('after train.csv')


# adapt validation frame_lists such that only the present videos are considered
ava_val_fl = pd.read_csv(os.path.join(orig_frame_lists, 'val.csv'), delimiter=' ')
unique_video_ids = ava_val[0].unique().tolist()

frames = []
for id in unique_video_ids:
    frames.append(ava_val_fl.loc[ava_val_fl['original_vido_id'] == id])

ava_val_fl = pd.concat(frames)
ava_val_fl = ava_val_fl.replace(np.nan, -1, regex=True)
ava_val_fl.to_csv(os.path.join(frame_list_dir, 'val.csv'), sep=' ', index=False)


print('after val.csv')




