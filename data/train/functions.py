import os

import os

from PIL import Image
from slowfast.visualization.video_visualizer import VideoVisualizer
from torchvision import transforms
import torch
import pandas as pd
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np




def ava_path(image_folder, video, sec):
    # load the image into torch tensor
    frame_id = (sec - 900) * 30
    filename = video + '_' + str(frame_id).zfill(6) + '.jpg'
    path = os.path.join(image_folder, video)
    path = os.path.join(path, filename)
    return path



def generate_detection_img(num_classes, class_name_json, video, second, detections_file, image_folder):
    # load the image into torch tensor
    path = ava_path(image_folder, video, second)

    # Initialize vide visualizer
    video_vis = VideoVisualizer(
            num_classes=num_classes,
            class_names_path=class_name_json,
            top_k=3,
            thres=0.7,
            lower_thres=0.3,
            common_class_names=None,
            colormap="Reds",
            mode="top-k",
        )

    # reads CHW
    img = Image.open(path)
    transformation = transforms.ToTensor()
    img = transformation(img)

    img = img*255
    img = img.type(torch.IntTensor)

    # switch to HWC
    img = img.permute(1,2,0)

    # Read predictions for the given frame and save content to preds and bboxes
    all_predictions = pd.read_csv(detections_file, header=None)

    # count number of video and frame occurance
    predictions = all_predictions.loc[(all_predictions[0] == video) & (all_predictions[1] == second)]

    num_boxes = int(len(predictions) / num_classes)
    num_boxes = max(num_boxes, 1)

    # iterate through through all boxes and create the predictions file
    preds = torch.zeros([num_boxes, num_classes])
    bboxes = torch.zeros([num_boxes, 4])

    predictions_reduced = predictions.drop(columns=[0,1,6])
    predictions_reduced = torch.tensor(predictions_reduced.values)


    for i in range(num_boxes):
        # access all the predictions of the i-th box
        preds[i,:] = predictions_reduced[i*num_classes:(i+1)*num_classes,4]
        bboxes[i,:] = predictions_reduced[i*num_classes,0:4]

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
    # outputs CHW
    output = output.transpose((2, 0, 1))



    return output



def generate_gt_img(num_classes, class_name_json, video, second, gt_file, image_folder):

    # load the image into torch tensor
    path = ava_path(image_folder, video, second)

    # Read predictions for the given frame and save content to preds and bboxes
    all_predictions = pd.read_csv(gt_file, header=None)

    # count number of video and frame occurance
    predictions = all_predictions.loc[(all_predictions[0] == video) & (all_predictions[1] == second)]


    # Initialize vide visualizer
    video_vis = VideoVisualizer(num_classes=num_classes,
                                class_names_path=class_name_json,
                                top_k=3,
                                thres=0.7,
                                lower_thres=0.3,
                                common_class_names=None,
                                colormap="Reds",
                                mode="thres", )

    # reads CHW
    img = Image.open(path)
    transformation = transforms.ToTensor()
    img = transformation(img)

    img = img * 255
    img = img.type(torch.IntTensor)

    # switch to HWC
    img = img.permute(1, 2, 0)



    # create the bboxes
    box_helper = predictions.drop(columns=[0, 1, 6, 7])
    unique_boxes = box_helper.drop_duplicates()
    bboxes = unique_boxes

    # fill the predictions
    num_boxes = unique_boxes.shape[0]
    preds = torch.zeros([num_boxes, num_classes])

    # read out the predictions for all boxes into lists
    p = 0
    for index, row in bboxes.iterrows():
        classes_base = all_predictions.loc[(all_predictions[2] == row[2]) & (all_predictions[3] == row[3]) & (all_predictions[4] == row[4]) & (all_predictions[5] == row[5])]
        classes = classes_base[6].tolist()
        for i in classes:
            preds[p, i - 1] = 1
        p += 1

    # create tensor from bboxes
    bboxes = torch.tensor(bboxes.values)

    # Scale the boxes according to image dimensions: (x,y) top left (x,y) bottom right
    height = img.shape[0]
    width = img.shape[1]

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height

    # bboxes = torch.tensor([[0.332*img.shape[1],0.336*img.shape[0],0.628*img.shape[1],0.676*img.shape[0]]])
    # bboxes = torch.tensor([[0.332,0.336,0.628,0.676]])

    output = video_vis.draw_one_frame(frame=img, preds=preds, bboxes=bboxes, alpha=0.5, text_alpha=0.7,
                                      ground_truth=False, )
    # CHW
    output = output.transpose((2, 0, 1))

    return output


def visualize_samples(samples, tensorboard, detections_file, gt_file, class_name_json, num_classes, image_folder,
                      iteration):

    # create a pool of samples to afterwards sample from
    file = pd.read_csv(detections_file, header=None)
    file_red = file[[0, 1]].drop_duplicates()

    # initialize empty image list to feed in the last
    image_list_gt = []
    image_list_detections = []

    # initialize empty image_name list
    name_gt = ''
    name_detections = ''

    # maximum height and width
    height_max = 0
    width_max = 0

    # iterate number of samples times
    for i in range(samples):
        # take corresponding sample
        x = file_red.sample()
        SEC = int(x[1].values[0])
        VIDEO = x[0].values[0]

        # DO THE WHOLE PROCESS FOR THE DETECTIONS
        output = generate_detection_img(num_classes=num_classes, class_name_json=class_name_json, video=VIDEO,
                                        second=SEC, detections_file=detections_file, image_folder=image_folder)

        height = output.shape[1]
        width = output.shape[2]

        if height > height_max:
            height_max = height
        if width > width_max:
            width_max = width

        name = VIDEO + '_' + str(SEC)
        name_detections = name_detections + '| ' + name + ' '
        image_list_detections.append(output)

        # DO THE WHOLE PROCESS FOR THE GROUNDTRUTH
        output = generate_gt_img(num_classes=num_classes, class_name_json=class_name_json, video=VIDEO, second=SEC,
                                 gt_file=gt_file, image_folder=image_folder)

        height = output.shape[1]
        width = output.shape[2]

        if height > height_max:
            height_max = height
        if width > width_max:
            width_max = width

        name = VIDEO + '_' + str(SEC)
        name_gt = name_gt + '| ' + name + ' '
        image_list_gt.append(output)

    # write to tensorboard for detections



    for i in range(len(image_list_detections)):
        image_list_detections[i] = np.pad(image_list_detections[i],
                                          ((0,0),
                                           (0,height_max-image_list_detections[i].shape[1]),
                                           (0,width_max-image_list_detections[i].shape[2])),
                                          'constant',
                                          constant_values=((0,0),(0,0),(0,0)))



    image_list_detections = np.stack(image_list_detections, axis=0)
    image_list_detections = torch.tensor(image_list_detections)

    grid = torchvision.utils.make_grid(image_list_detections, nrow=samples)
    writer = SummaryWriter(log_dir=tensorboard + str(iteration) + '_detections')
    writer.add_image(name_detections, grid, 0)
    writer.close()

    # write to tensorboard for gt
    for i in range(len(image_list_gt)):
        image_list_gt[i] = np.pad(image_list_gt[i],
                                          ((0,0),
                                           (0,height_max-image_list_gt[i].shape[1]),
                                           (0,width_max-image_list_gt[i].shape[2])),
                                          'constant',
                                          constant_values=((0,0),(0,0),(0,0)))


    image_list_gt = np.stack(image_list_gt, axis=0)
    image_list_gt = torch.tensor(image_list_gt)

    grid = torchvision.utils.make_grid(image_list_gt, nrow=samples)
    writer = SummaryWriter(log_dir=tensorboard + str(iteration) + '_gt')
    writer.add_image(name_gt, grid, 0)
    writer.close()