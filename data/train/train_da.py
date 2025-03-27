#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import random
import numpy as np
import pprint
import torch
import os
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TrainMeter, ValMeter, DA_AVAMeter
from slowfast.utils.multigrid import MultigridSchedule


logger = logging.get_logger(__name__)


def train_epoch(
    da_train_loader, train_loader, model, optimizer, optimizer_aux, train_meter, cur_epoch, cfg, writer=None
):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    da_train_iterator = iter(da_train_loader)

    # if train_loader is larger than da_train_loader, add a backup data loader
    print('Length of train_loader: ', data_size)
    print('Length of da_loader: ', len(da_train_loader))


    if data_size > len(da_train_loader):
        da_train_iterator2 = iter(da_train_loader)
        print('Created a backup da_loader')
        #data_size = len(da_train_loader)

    # initialze lists for the confusion matrices
    y_true = []
    y_pred = []
    y_true_interim = []
    y_pred_interim = []
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, 'confusion')):
        os.mkdir(os.path.join(cfg.OUTPUT_DIR, 'confusion'))

    #ct = 0
    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        """
        if cur_iter == (data_size - 1):
            print('meta: ', meta)
            #print('last 4 model layers: ', list(model.parameters())[-4:-2])
            #print('all model weights: ', list(model.parameters())[606:-2])
            layer = 0
            for name, p in model.named_parameters():
                if layer > cfg.MODEL.FREEZE_TO:
                    print(layer, name, p)
                layer += 1
        """

        """
        ct += 1
        if ct == data_size:
            continue
        """

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)


        if cur_iter < len(da_train_loader):
            (inputs2, labels2, _, meta2) = next(da_train_iterator)
            print(cur_iter, ' taking samples from da_train_loader')
        else:
            (inputs2, labels2, _, meta2) = next(da_train_iterator2)
            print(cur_iter, ' taking samples from da_train_loader2')

        #(inputs2, labels2, _, meta2) = next(da_train_iterator)

        if cfg.NUM_GPUS:
            if isinstance(inputs2, (list,)):
                for i in range(len(inputs2)):
                    inputs2[i] = inputs2[i].cuda(non_blocking=True)
            else:
                inputs2 = inputs2.cuda(non_blocking=True)
            labels2 = labels2.cuda()
            for key, val in meta2.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta2[key] = val.cuda(non_blocking=True)


        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        lr_aux = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg) / cfg.DA.LR_FACTOR
        optim.set_lr(optimizer_aux, lr_aux)

        train_meter.data_toc()

        if cfg.DETECTION.ENABLE:

            """
            activation = {}

            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach().cpu()
                return hook

            # register all the wanted hooks for one forward pass
            model.s5.pathway0_res0.branch1_bn.register_forward_hook(get_activation('s5.pathway0_res0.branch1_bn'))
            model.s5.pathway0_res0.branch2.register_forward_hook(get_activation('s5.pathway0_res0.branch2'))
            model.s5.pathway0_res1.branch2.register_forward_hook(get_activation('s5.pathway0_res1.branch2'))
            model.s5.pathway0_res2.branch2.register_forward_hook(get_activation('s5.pathway0_res2.branch2'))

            model.s5.pathway1_res0.branch1_bn.register_forward_hook(get_activation('s5.pathway1_res0.branch1_bn'))
            model.s5.pathway1_res0.branch2.register_forward_hook(get_activation('s5.pathway1_res0.branch2'))
            model.s5.pathway1_res1.branch2.register_forward_hook(get_activation('s5.pathway1_res1.branch2'))
            model.s5.pathway1_res2.branch2.register_forward_hook(get_activation('s5.pathway1_res2.branch2'))

            model.head.act.register_forward_hook(get_activation('head.act'))
            """

            preds, _ = model(inputs, meta["boxes"])

            """
            import pickle
            with open('/srv/beegfs02/scratch/da_action/data/output/randomness/run2/' + str(cur_iter) +
                      'features.pkl', 'wb') as f:
                pickle.dump(activation, f)


            print(str(cur_iter), 'predictions: ', activation['head.act'])
            """

            """
            print(activation['s5.pathway1_res2.branch2.c_bn'][2,2,2,2,:])
            print(activation['s5.pathway1_res2.branch2'][2,2,2,2,:])
            """



            """
            if cur_iter == (data_size - 1):
                print('predictions 1: ', preds)
            """




            _, preds2 = model(inputs2, meta2["boxes"])
        else:
            preds = model(inputs)
        # Explicitly declare reduction to mean.

        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        loss_fun2 = losses.get_loss_func("cross_entropy")(reduction="mean")
        labels2 = labels2.long()
        labels2 = torch.argmax(labels2, dim=1)

        # append predictions and ground truth to list
        for i in range(labels2.shape[0]):
            y_true.append(int(labels2[i]))
            y_pred.append(int(torch.argmax(preds2[i, :])))
            y_true_interim.append(int(labels2[i]))
            y_pred_interim.append(int(torch.argmax(preds2[i, :])))

        # Luca: Cross Entropy expects .long() in the second argument
        """
        if cfg.MODEL.LOSS_FUNC == 'cross_entropy':
            labels = labels.long()
            labels = torch.argmax(labels, dim=1)
        """

        # Compute the loss.
        loss1 = loss_fun(preds, labels)


        #print(str(cur_iter) + 'loss1 when other has 0 weight: ', loss1)




        #loss = (cfg.DA.WEIGHT_MAIN * loss1) + (cfg.DA.WEIGHT_AUX * loss2)

        # check Nan Loss.
        misc.check_nan_losses(loss1)

        # Perform the backward pass.



        optimizer.zero_grad()

        # Set random seed from configs: https://datascience.stackexchange.com/questions/56614/why-would-pytorch-be
        # -non-deterministic-for-batch-only
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        #torch.set_deterministic(True)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        """
        import pickle
        with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss1/run1/loss_before_bw.pkl',
                  'wb') as f:
            pickle.dump(loss.detach().cpu(), f)

        with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss1/run1/model_before_bw.pkl', 'wb') as f:
            pickle.dump(model, f)

        """
        loss1.backward() #computes gradients
        """
        with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss1/run1/loss_after_bw.pkl',
                  'wb') as f:
            pickle.dump(loss.detach().cpu(), f)

        with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss1/run1/model_after_bw.pkl', 'wb') as f:
            pickle.dump(model, f)
        """
        # Update the parameters.
        optimizer.step()

        """
        with open('/srv/beegfs02/scratch/da_action/data/output/first_it_loss1/run1/model_optimizer.pkl',
                  'wb') as f:
            pickle.dump(model, f)
        import pdb
        pdb.set_trace()
        """

        loss2 = loss_fun2(preds2, labels2)
        misc.check_nan_losses(loss2)
        optimizer_aux.zero_grad()
        # Set random seed from configs: https://datascience.stackexchange.com/questions/56614/why-would-pytorch-be
        # -non-deterministic-for-batch-only
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        # torch.set_deterministic(True)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        loss2.backward()
        optimizer_aux.step()


        train_meter.iter_toc()  # do not measure allreduce for this meter


        loss = loss1

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {"Train/loss_class": loss1, "Train/loss_rot": loss2, "Train/lr": lr, "Train/lr_aux": lr_aux},
                    global_step=data_size * cur_epoch + cur_iter,
                )
                """
                writer.add_scalars(
                    {"Train/loss_total": loss, "Train/loss_class": loss1, "Train/loss_rot": loss2, "Train/lr": lr},
                    global_step=data_size * cur_epoch + cur_iter, )
                """

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                writer.add_scalars(
                    {
                        "Train/loss": loss,
                        "Train/lr": lr,
                        "Train/Top1_err": top1_err,
                        "Train/Top5_err": top5_err,
                    },
                    global_step=data_size * cur_epoch + cur_iter,
                )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

        # save confusion matrix during epoch, in case it is wanted
        if cfg.DA.F_CONFUSION != -1 and cur_iter % cfg.DA.F_CONFUSION == 0:  # was 500 previously (also once it was 100)

            cm = misc.create_cm(y_true_interim, y_pred_interim, 4)
            misc.plot_confusion_matrix(cm=cm, normalize=False, target_names=['0', '90ccw', '180ccw', '270ccw'],
                                       title="Confusion Matrix Epoch " + str(cur_epoch) + ", Batch " + str(cur_iter),
                                       file=cfg.OUTPUT_DIR + '/confusion/confusion' + str(cur_epoch) + '_' + str(
                                           cur_iter) + '.jpg', tensorboard=cfg.OUTPUT_DIR, epoch=cur_epoch,
                                       cur_iter=cur_iter)
            y_true_interim.clear()
            y_pred_interim.clear()

            # save the model (always overwrite)
            index = cur_iter / cfg.DA.F_CONFUSION  # was 500 previously
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, int(100 + cur_epoch * 100 + index), cfg)




    # create and save confusion matrix for current epoch
    # print(len(y_true))
    cm = misc.create_cm(y_true, y_pred, 4)
    misc.plot_confusion_matrix(cm=cm, normalize=False, target_names=['0', '90ccw', '180ccw', '270ccw'],
                            title="Confusion Matrix Epoch " + str(cur_epoch),
                            file=cfg.OUTPUT_DIR + '/confusion/confusion' + str(cur_epoch) + '.jpg',
                            tensorboard=cfg.OUTPUT_DIR, epoch=cur_epoch)
    y_true.clear()
    y_pred.clear()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None, writer_str="Val/mAP"):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds, _ = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]
            """
            # Determine the validation loss
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Luca: Cross Entropy expects .long() in the second argument
            if cfg.MODEL.LOSS_FUNC == 'cross_entropy':
                labels = labels.long()
                labels = torch.argmax(labels, dim=1)

            # Compute the loss.
            loss = loss_fun(preds, labels)
            # check Nan Loss.
            misc.check_nan_losses(loss)
            """

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)



        else:
            preds = model(inputs)

            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    writer.add_scalars(
                        {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                        global_step=len(val_loader) * cur_epoch + cur_iter,
                    )

            val_meter.update_predictions(preds, labels)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)

    # Luca Sieber: log images to tensorboard output for each validation iteration
    #TODO write code that if set writes some sample images to tensorboard
    #TODO set the evaluate_ava in ava_eval_helper.py accordingly to save the files at the right place


    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            #writer.add_scalars(
            #    {"Val/mAP": val_meter.full_map, "Val/loss": loss}, global_step=cur_epoch
            #)

            writer.add_scalars({writer_str: val_meter.full_map}, global_step=cur_epoch)
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train_da(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Set up environment.
    du.init_distributed_training(cfg)

    # Set random seed from configs.  https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    #torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    optimizer_aux = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    #start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)


    # Luca Sieber
    if cfg.MODEL.FREEZE_TO >= 0:
        layer = 0
        for name, p in model.named_parameters():
            if layer <= cfg.MODEL.FREEZE_TO:
                p.requires_grad = False
                print(layer, "frozen", name)
            else:
                print(layer, "unfrozen", name)

            layer += 1
        start_epoch = cu.load_train_checkpoint(cfg=cfg, model=model, optimizer=None)
    else:
        start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Luca Sieber activate this section if a detailed layer wise count shall be returned
    """
    counter = 0
    layer = 0
    for name, p in model.named_parameters():
        count = p.numel()
        counter += count
        print(layer, ' ', name, 'number of parameters: ', count, ' accumulated: ', counter)
        layer += 1
    """



    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create the video train and val loaders for da
    da_train_loader = loader.construct_loader_da(cfg, "da_train")
    #da_val_loader = loader.construct_loader_da(cfg, "da_val")

    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )


    # Create meters.
    if cfg.DETECTION.ENABLE:
        # construct meters
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")

        # construct da meter
        #da_val_meter = DA_AVAMeter(len(da_val_loader), cfg, mode="da_val")

    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):


        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        train_epoch(da_train_loader, train_loader, model, optimizer, optimizer_aux, train_meter, cur_epoch, cfg, writer)

        is_checkp_epoch = (
            cu.is_checkpoint_epoch(
                cfg,
                cur_epoch,
                None if multigrid is None else multigrid.schedule,
            )
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
            (is_checkp_epoch or is_eval_epoch)
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer, "Val/AVA-mAP")
            #eval_epoch(da_val_loader, model, da_val_meter, cur_epoch, cfg, writer, "Val/Kinetics-mAP")

    if writer is not None:
        writer.close()
