#TODO sort order of samples in files, such that single frames are together -> already done in create_files.py
#TODO allow to have non-consecutive classes with labels being correctly traansferred -> already done in create_files.py
#TODO include option for background class when training in another scheme.
"this script should have become a version that includes a background class, but is not yet finished"

import os
import shutil
import numpy as np
from functions import reduce_pbtxt
import pandas as pd

orig_annot = '/srv/beegfs02/scratch/da_action/data/ava/annotations_v2.2'
orig_frame_lists = '/srv/beegfs02/scratch/da_action/data/ava/frame_lists'

#############################
# Hyperparameters
work_dir = '/srv/beegfs02/scratch/da_action/data/ava'

SAMPLES_TRAIN = 8
SAMPLES_VAL = 3

CLASSES = [1,2,3,4,5]

##############################
# generate config name
config_name = str(len(CLASSES)) + '_' + str(SAMPLES_TRAIN) + '_' + str(SAMPLES_VAL) + '_' 'BG'

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
#TODO add the background class to the .pbtxt
reduce_pbtxt(orig_annot=orig_annot, annot_dir=annot_dir, classes=CLASSES)
"""
original = os.path.join(orig_annot, 'ava_action_list_v2.2.pbtxt')
target = os.path.join(annot_dir, 'ava_action_list_v2.2.pbtxt')
shutil.copyfile(original, target)
"""

# adapt classes such that it is indexed from 1 ...
#CLASSES = [x + 1 for x in CLASSES]

# copy ava_included_timestamps_v2.2.txt
original = os.path.join(orig_annot, 'ava_included_timestamps_v2.2.txt')
target = os.path.join(annot_dir, 'ava_included_timestamps_v2.2.txt')
shutil.copyfile(original, target)

# TRAINING
# copy ava_train_excluded_timestamps_v2.2.csv
original = os.path.join(orig_annot, 'ava_train_excluded_timestamps_v2.2.csv')
target = os.path.join(annot_dir, 'ava_train_excluded_timestamps_v2.2.csv')
shutil.copyfile(original, target)

# load ava_train_v2.2.csv
# think of not loading the entire file but only a certain number of lines
ava_train = pd.read_csv(os.path.join(orig_annot, 'ava_train_v2.2.csv'), header=None, nrows=160*100*SAMPLES_TRAIN)
#ava_train = pd.read_csv(os.path.join(orig_annot, 'ava_train_v2.2.csv'), header=None)

# remove the excluded timestamps
#IT SEEMS THAT THE NONVALID TIMESTAMPS HAVE ALREADY BEEN EXCLUDED (OR THEY ARE ANYWAY NOT PART OF THE ANNOTATION FILES)
ava_train_excl = pd.read_csv(os.path.join(orig_annot, 'ava_train_excluded_timestamps_v2.2.csv'), header=None)
for index, row in ava_train_excl.iterrows():
    indexNames = ava_train[(ava_train[0] == row[0]) & (ava_train[1] == row[1])].index
    ava_train.drop(indexNames, inplace=True)


# extract the number of wanted samples from the file
frames = []
# iterate through list of wanted classes
for cl in CLASSES:
    frames.append(ava_train.loc[ava_train[6] == cl].head(SAMPLES_TRAIN))
ava_train_current = pd.concat(frames)

#TODO add the remaining predictions from used frames (idea: simply add all samples from touched frames again and
# then remove duplicates to end up with (need to test this properly, after adding modify the class to background index)

# extract list of unique (video_id, time) tuples
tuples = []
for index, row in ava_train.iterrows():
    tuples.append((row[0], row[1]))

# eliminate double appearances of (video_id, time) occurances
tuples = list(set(tuples))

# extract all these (video_id, time) combinations from the original ava_train dictionary
frames = []


print(tuples[0])

ava_train = ava_train.loc[ava_train[0] == tuples[0][0]]
print(ava_train)
ava_train = ava_train.loc[ava_train[1] == tuples[0][1]]
print(ava_train)
#TODO this file is work in progress -> probably adding the background class is anyway not necessary when using a one
# vs all classifier ->
"""
for t in tuples:
    frames.append(ava_train.loc[ava_train[0] == t[0] & ava_train[1] == t[1]].head(1000))

"""

"""

# sort for the first two columns
ava_train = ava_train.sort_values(by=[0,1])

# save to a fi
ava_train.to_csv(os.path.join(annot_dir, 'ava_train_v2.2.csv'), header=None, index=False, float_format='%.3f')

# load ava_train_predicted_boxes.csv even if it is probably not used in actual project
ava_train_pred = pd.read_csv(os.path.join(orig_annot, 'ava_train_predicted_boxes.csv'), header=None,
                             nrows=80*150*SAMPLES_TRAIN)
frames = []
for index, row in ava_train.iterrows():
    frames.append(ava_train_pred.loc[(ava_train_pred[0] == row[0]) & (ava_train_pred[1] == row[1])])
ava_train_pred = pd.concat(frames)

# save to a file
ava_train_pred.to_csv(os.path.join(annot_dir, 'ava_train_predicted_boxes.csv'), header=None,
                      index=False, float_format='%.3f')



# VALIDATION
# copy ava_train_excluded_timestamps_v2.2.csv
original = os.path.join(orig_annot, 'ava_val_excluded_timestamps_v2.2.csv')
target = os.path.join(annot_dir, 'ava_val_excluded_timestamps_v2.2.csv')
shutil.copyfile(original, target)

# load ava_val_v2.2.csv
# think of not loading the entire file but only a certain number of lines
#ava_val = pd.read_csv(os.path.join(orig_annot, 'ava_val_v2.2.csv'), header=None, nrows=80*100*SAMPLES_VAL)
ava_val = pd.read_csv(os.path.join(orig_annot, 'ava_val_v2.2.csv'), header=None)

# remove the excluded timestamps
#IT SEEMS THAT THE NONVALID TIMESTAMPS HAVE ALREADY BEEN EXCLUDED (OR THEY ARE ANYWAY NOT PART OF THE ANNOTATION FILES)
ava_val_excl = pd.read_csv(os.path.join(orig_annot, 'ava_val_excluded_timestamps_v2.2.csv'), header=None)
for index, row in ava_val_excl.iterrows():
    indexNames = ava_val[(ava_val[0] == row[0]) & (ava_val[1] == row[1])].index
    ava_val.drop(indexNames, inplace=True)

# extract the number of wanted samples from the file
frames = []
# iterate through list of wanted classes
for cl in CLASSES:
    frames.append(ava_val.loc[ava_val[6] == cl].head(SAMPLES_VAL))
ava_val = pd.concat(frames)

#TODO add the remaining predictions from used frames (idea: simply add all samples from touched frames again and
# then remove duplicates to end up with (need to test this properly, after adding modify the class to background index)

# sort for the first two columns
ava_val = ava_val.sort_values(by=[0,1])

# save to a file
ava_val.to_csv(os.path.join(annot_dir, 'ava_val_v2.2.csv'), header=None, index=False, float_format='%.3f')


#load ava_val_predicted_boxes.csv -> this is definitely used in the classification
#ava_val_pred = pd.read_csv(os.path.join(orig_annot, 'ava_val_predicted_boxes.csv'), header=None,
# nrows=80*150*SAMPLES_VAL)
ava_val_pred = pd.read_csv(os.path.join(orig_annot, 'ava_val_predicted_boxes.csv'), header=None)

frames = []
for index, row in ava_val.iterrows():
    frames.append(ava_val_pred.loc[(ava_val_pred[0] == row[0]) & (ava_val_pred[1] == row[1])])
ava_val_pred = pd.concat(frames)

# save to a file
ava_val_pred.to_csv(os.path.join(annot_dir, 'ava_val_predicted_boxes.csv'), header=None,
                      index=False, float_format='%.3f')




# adapt train frame_lists such that only the present videos are considered
ava_train_fl = pd.read_csv(os.path.join(orig_frame_lists, 'train.csv'), delimiter=' ')
unique_video_ids = ava_train[0].unique().tolist()

frames = []
for id in unique_video_ids:
    frames.append(ava_train_fl.loc[ava_train_fl['original_vido_id'] == id])

ava_train_fl = pd.concat(frames)
ava_train_fl = ava_train_fl.replace(np.nan, -1, regex=True)
ava_train_fl.to_csv(os.path.join(frame_list_dir, 'train.csv'), sep=' ', index=False)




# adapt validation frame_lists such that only the present videos are considered
ava_val_fl = pd.read_csv(os.path.join(orig_frame_lists, 'val.csv'), delimiter=' ')
unique_video_ids = ava_val[0].unique().tolist()

frames = []
for id in unique_video_ids:
    frames.append(ava_val_fl.loc[ava_val_fl['original_vido_id'] == id])

ava_val_fl = pd.concat(frames)
ava_val_fl = ava_val_fl.replace(np.nan, -1, regex=True)
ava_val_fl.to_csv(os.path.join(frame_list_dir, 'val.csv'), sep=' ', index=False)




"""



