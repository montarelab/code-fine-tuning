# add new folder to path since other modules are used
import sys
sys.path.insert(0, "/home/sieberl/SA2020/pyslowfast/slowfast")

from datetime import datetime
import os
import torch
import pandas as pd
from PIL import Image
from slowfast.visualization.video_visualizer import VideoVisualizer
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import cv2

# Hyperparameters
SEC = 90
VIDEO = '02pQ-4vI7-I'
LOG_DIR = 'train_detections6'

TENSORBOARD = "/srv/beegfs02/scratch/da_action/data/output/tb_visualization/" #done
#TENSORBOARD = "/srv/beegfs02/scratch/da_action/data/output/ex_5_200_40_v2/"

GROUNDTRUTH = True
results_file = '/home/sieberl/SA2020/pyslowfast/experiments/ex_5_200_40_v2/39_groundtruth_latest.csv'
results_file = '/home/sieberl/SA2020/pyslowfast/get_datasets/kinetics/detections.csv'

class_name_json = '/srv/beegfs02/scratch/da_action/data/ava/80_annotations_10_500_100/class_names.json'
NUM_CLASSES = 80

image_folder = '/srv/beegfs02/scratch/da_action/data/kinetics700/frames'
ava = False

# The number of predictions shown per frame depends on whether it is groundtruth or not
if GROUNDTRUTH:
    top_k = 3
else:
    top_k = 3

# Initialize vide visualizer
video_vis = VideoVisualizer(
        num_classes=5,
        class_names_path=class_name_json,
        top_k=top_k,
        thres=0.7,
        lower_thres=0.3,
        common_class_names=None,
        colormap="Oranges",
        mode="top-k",
    )

# load the image into torch tensor
if ava:
    frame_id = (SEC - 900) * 30
else:
    frame_id = SEC

filename = VIDEO + '_' + str(frame_id).zfill(6) + '.jpg'
path = os.path.join(image_folder, VIDEO)
path = os.path.join(path, filename)

# reads CHW
img = Image.open(path)
transformation = transforms.ToTensor()
img = transformation(img)

img = img*255
img = img.type(torch.IntTensor)

# switch to HWC
img = img.permute(1,2,0)

# Read predictions for the given frame and save content to preds and bboxes
all_predictions = pd.read_csv(results_file, header=None)

# count number of video and frame occurance
predictions = all_predictions.loc[(all_predictions[0] == VIDEO) & (all_predictions[1] == SEC)]

if GROUNDTRUTH:

    # create the bboxes
    box_helper = predictions.drop(columns=[0, 1, 6, 7])
    unique_boxes = box_helper.drop_duplicates()
    bboxes = unique_boxes

    # fill the predictions
    num_boxes = unique_boxes.shape[0]
    preds = torch.zeros([num_boxes, NUM_CLASSES])

    # read out the predictions for all boxes into lists
    p = 0
    for index, row in bboxes.iterrows():
        classes_base = all_predictions.loc[(all_predictions[2] == row[2]) & (all_predictions[3] == row[3]) & (
                all_predictions[4] == row[4]) & (all_predictions[5] == row[5])]
        classes = classes_base[6].tolist()
        print(index)
        for i in classes:
            preds[p, i-1] = 1

        p += 1

    # create tensor from bboxes
    bboxes = torch.tensor(bboxes.values)

else:
    num_boxes = int(len(predictions) / NUM_CLASSES)
    num_boxes = max(num_boxes, 1)

    # iterate through through all boxes and create the predictions file
    preds = torch.zeros([num_boxes, NUM_CLASSES])
    bboxes = torch.zeros([num_boxes, 4])

    predictions_reduced = predictions.drop(columns=[0,1,6])
    predictions_reduced = torch.tensor(predictions_reduced.values)


    for i in range(num_boxes):
        # access all the predictions of the i-th box
        preds[i,:] = predictions_reduced[i*NUM_CLASSES:(i+1)*NUM_CLASSES,4]
        bboxes[i,:] = predictions_reduced[i*NUM_CLASSES,0:4]



"""
# create predictions
#TODO write code that gathers predictions automatically for a given frame and video
preds = torch.tensor([[0.9547, 0.0004, 0.0206, 0.0001, 0.0049]])

# create bounding boxes
#TODO write code that gathers boxes automatically for a given frame and video
bboxes = torch.tensor([[0.055,0.300,0.868,0.972]])
"""

# Scale the boxes according to image dimensions: (x,y) top left (x,y) bottom right
height = img.shape[0]
width = img.shape[1]

bboxes[:,[0,2]] = bboxes[:,[0,2]] * width
bboxes[:,[1,3]] = bboxes[:,[1,3]] * height


#bboxes = torch.tensor([[0.332*img.shape[1],0.336*img.shape[0],0.628*img.shape[1],0.676*img.shape[0]]])
#bboxes = torch.tensor([[0.332,0.336,0.628,0.676]])


output = video_vis.draw_one_frame(
        frame=img,
        preds=preds,
        bboxes=bboxes,
        alpha=0.5,
        text_alpha=0.7,
        ground_truth=False,
    )

#logdir = TENSORBOARD + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = TENSORBOARD + datetime.now().strftime(LOG_DIR)

output = output.transpose((2,0,1))

writer = SummaryWriter(log_dir=logdir)
writer.add_image('images', output, 0)

writer.close()
#cv2.imwrite('/home/sieberl/SA2020/test.jpg', output)