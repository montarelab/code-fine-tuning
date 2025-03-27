#https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=dq9GY37ml1kr

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import pandas as pd
import csv

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

kinetics = '/srv/beegfs02/scratch/da_action/data/kinetics700/frames'
#source = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_v2.1.csv'
#source = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_v2.1.csv'
source = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_v2.1.csv'
#target_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_predicted_boxes.csv'
#target_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_predicted_boxes.csv'
target_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_predicted_boxes.csv'
#failed_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_val_failed_path.csv'
#failed_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_failed_path.csv'
failed_file = '/srv/beegfs02/scratch/da_action/data/kinetics700/annotations/kinetics_train_failed_path.csv'

# Function declaration
def predict_boxes(path):

    # (height, width, channels)
    im = cv2.imread(path)

    cfg = get_cfg()

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # loads model id 137849600 from: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # boxes are given as [x1,y1,x2,y2] where 1 is the top left point and 2 is the bottom right point of the detection

    # for all extracted information, only use human detections
    classes = outputs["instances"].pred_classes.cpu().numpy()

    human_index = np.where(classes == 0)[0]

    boxes = outputs["instances"].pred_boxes.tensor[human_index,:]

    outputs["instances"].pred_boxes.tensor = boxes

    outputs["instances"].pred_classes = outputs["instances"].pred_classes[human_index]

    scores = outputs["instances"].scores[human_index]
    outputs["instances"].scores = scores


    outputs["instances"].pred_masks = outputs["instances"].pred_masks[human_index,:,:]

    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imwrite('/home/sieberl/SA2020/test.jpg', out.get_image()[:, :, ::-1])
    #cv2_imshow(out.get_image()[:, :, ::-1])

    """
    im_width = out.get_image()[:, :, ::-1].shape[1]
    im_height = out.get_image()[:, :, ::-1].shape[0]
    """
    im_width = im.shape[1]
    im_height = im.shape[0]


    boxes[:,0] = boxes[:,0] / im_width
    boxes[:,2] = boxes[:,2] / im_width

    boxes[:,1] = boxes[:,1] / im_height
    boxes[:,3] = boxes[:,3] / im_height

    #boxes = boxes.type(torch.FloatTensor)
    return boxes, scores



# Start of the actual script

# load a list of the video names where we need to create predictions
source = pd.read_csv(source, header=None, sep=',')
source = source.drop([2,3,4,5,6], axis=1)
source = source.drop_duplicates()
print('after dropping duplicates')

failed_paths = []

"""
# for debugging
video_name = '_0DJsE0De7M'
frame_finder = video_name + '/' + video_name + '_' + str(90).zfill(6) + '.jpg'

path = os.path.join(kinetics, frame_finder)
boxes, scores = predict_boxes(path)
print('boxes after processing: ', boxes)
"""
print('total job size: ', len(source[0]))
counter = 0
# open the target file
f = open(target_file, 'w')
with f:
    writer = csv.writer(f)

    for index, row in source.iterrows():

        # calculate the frame id:
        add = int(30 * (row[1] % 1))
        frame_id = int(90 + add)
        print(frame_id)
        
        video_name = row[0]
        frame_finder = video_name + '/' + video_name + '_' + str(frame_id).zfill(6) + '.jpg'
        path = os.path.join(kinetics, frame_finder)

        print('current counter: ', counter)
        counter += 1

        if os.path.isfile(path):
            print('successful: ', frame_finder)
            boxes, scores = predict_boxes(path)

            # write predictions to a .csv file
            # data to be written row-wise in csv fil
            list = []
            for i in range(boxes.shape[0]):
                list.append([video_name, row[1], "%.3f" % boxes[i, 0].item(), "%.3f" % boxes[i, 1].item(),
                             "%.3f" % boxes[i, 2].item(), "%.3f" % boxes[i, 3].item(), '', "%.6f" % scores[i].item()])
            for row in list:
                writer.writerow(row)

        else:
            failed_paths.append(path)
            print('failed: ', frame_finder)
            
        


f = open(failed_file, 'w')
with f:
    writer = csv.writer(f)
    for row in failed_paths:
        writer.writerow(row)



















