import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.cga import DeepLab
from nets.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
Important notes for training your own semantic segmentation model:

1. Carefully check your dataset format before training. This library requires VOC format.
   Input images should be .jpg (auto-resized before training, any size accepted).
   Grayscale images are automatically converted to RGB.
   If your images don't end with .jpg, batch-convert them to .jpg first.

   Labels should be .png (auto-resized before training).
   Important: Each pixel value in the label indicates the class of that pixel.
   Common online datasets use 0 for background and 255 for objects — these will train
   but produce no meaningful predictions. Change to: 0 for background, 1 for object, etc.
   Pixel values must be contiguous class indices starting at 0.

2. Loss values indicate convergence. Watch for a downward trend in validation loss.
   If validation loss plateaus, the model has converged. The absolute loss value is
   less important than the trend. Loss curves are saved in logs/loss_YYYY_MM_DD_HH_MM_SS/.

3. Trained weights are saved in the logs/ folder. Each Epoch contains multiple Steps.
   Each Step performs one gradient descent. Weights are only saved after full epochs,
   not individual steps.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda    Whether to use Cuda.
    #           Set to False if no GPU is available.
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed    Fix random seed for reproducibility
    #           across independent runs.
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed     Whether to use single-machine multi-GPU distributed
    #                   training. Terminal commands only support Ubuntu.
    #                   CUDA_VISIBLE_DEVICES is used to specify GPUs on Ubuntu.
    #                   Windows defaults to DataParallel (DP) mode, no DDP.
    #   DP mode:
    #       Set             distributed = False
    #       Terminal:       CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP mode:
    #       Set             distributed = True
    #       Terminal:       CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn     Whether to use synchronized batch normalization,
    #               available in DDP multi-GPU mode.
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16        Whether to use mixed precision training.
    #               Reduces memory usage by ~50%. Requires PyTorch >= 1.7.1.
    #---------------------------------------------------------------------#
    fp16            = False
    #-----------------------------------------------------#
    #   num_classes     Must modify for your own dataset.
    #                   Number of classes + 1 (e.g., 2+1=3).
    #-----------------------------------------------------#
    num_classes     = 8 #lovedata
    num_classes     = 7 #potsdam
    # num_classes     = 3 #mydata
    #---------------------------------#
    #   Backbone network:
    #   mobilenet
    #   xception
    #---------------------------------#
    backbone        = "mobilenet"

    pretrained      = False

    model_path      = "model_data/deeplab_mobilenetv2.pth"
    #---------------------------------------------------------#
    #   downsample_factor   Downsampling factor: 8 or 16.
    #                       8 gives smaller downsampling and
    #                       theoretically better results, but
    #                       requires more GPU memory.
    #---------------------------------------------------------#
    downsample_factor   = 16
    #------------------------------#
    #   Input image size
    #------------------------------#
    # input_shape         = [740, 740]
    input_shape         = [512, 512]
    # input_shape         = [640, 640]
    # input_shape         = [850, 850]



    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 0
    #------------------------------------------------------------------#
    #   Unfreeze phase training parameters.
    #   At this stage the backbone is no longer frozen; the feature
    #   extraction network will be updated. Uses more GPU memory.
    #   UnFreeze_Epoch          Total training epochs
    #   Unfreeze_batch_size     Batch size after unfreezing
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 16
    #------------------------------------------------------------------#
    #   Freeze_Train    Whether to use freeze training.
    #                   Default: freeze backbone first, then unfreeze.
    #------------------------------------------------------------------#
    Freeze_Train        = False

    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, LR schedule
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         Maximum learning rate.
    #                   Adam optimizer: Init_lr=5e-4 recommended.
    #                   SGD optimizer:  Init_lr=7e-3 recommended.
    #   Min_lr          Minimum learning rate, defaults to 0.01 * Init_lr.
    #------------------------------------------------------------------#
    Init_lr             = 7e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  Optimizer type: 'adam' or 'sgd'.
    #                   Adam: Init_lr=5e-4 recommended.
    #                   SGD:  Init_lr=7e-3 recommended.
    #   momentum        Momentum parameter for optimizer.
    #   weight_decay    Weight decay to prevent overfitting.
    #                   Adam may handle weight_decay incorrectly; set to 0.
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 1e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   Learning rate decay: 'step' or 'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     Save weights every N epochs
    #------------------------------------------------------------------#
    save_period         = 5
    #------------------------------------------------------------------#
    #   save_dir        Directory for saving weights and log files
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag       Whether to evaluate on validation set during training
    #   eval_period     Evaluate every N epochs. Frequent evaluation is slow
    #                   and will significantly slow down training.
    #   The mIoU here may differ from get_miou.py because:
    #   (1) This evaluates on the validation set.
    #   (2) Evaluation parameters here are conservative for speed.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 5

    #------------------------------------------------------------------#
    #   VOCdevkit_path  Path to the dataset
    #------------------------------------------------------------------#
    VOCdevkit_path  = '../datasets/VOCdevkit_Potsdam'
    #------------------------------------------------------------------#
    #   Recommended settings:
    #   Few classes:        set to True.
    #   Many classes (>10): set to True if batch_size > 10, else False.
    #------------------------------------------------------------------#
    dice_loss       = True
    #------------------------------------------------------------------#
    #   Whether to use focal loss to address class imbalance
    #------------------------------------------------------------------#
    focal_loss      = True
    #------------------------------------------------------------------#
    #   Whether to assign different loss weights to each class.
    #   Default is balanced. If set, use a numpy array with length
    #   equal to num_classes, e.g.:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     Number of data loading threads. 1 = single thread.
    #                   Multi-threading speeds up data loading but uses
    #                   more memory. Enable when I/O is the bottleneck
    #                   (GPU is much faster than disk read speed).
    #------------------------------------------------------------------#
    num_workers         = 4

    seed_everything(seed)
    #------------------------------------------------------#
    #   Configure GPUs to use
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #----------------------------------------------------#
    #   Download pretrained weights
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    model   = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   See README for weight file download instructions.
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        #------------------------------------------------------#
        #   Load weights by matching keys between pretrained
        #   weights and model state dict.
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Show keys that did not match
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "...\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "...\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m[Info] Head layers failing to load is normal. Backbone layers failing to load indicates an error.\033[0m")

    #----------------------#
    #   Record Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    #------------------------------------------------------------------#
    #   torch 1.2 does not support amp; use torch >= 1.7.1 for fp16.
    #   Therefore "could not be resolve" may appear on torch 1.2.
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   Multi-GPU Sync BatchNorm
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   Multi-GPU parallel
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #---------------------------#
    #   Read dataset txt files
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #---------------------------------------------------------#
        #   Total training epochs = number of passes through the entire dataset.
        #   Total training steps = number of gradient descent steps.
        #   Each epoch contains multiple steps, each step does one gradient descent.
        #   This only suggests a minimum epoch count; only the unfreeze phase is counted.
        #----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('Dataset too small for training. Please expand the dataset.')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] When using the %s optimizer, set total training steps above %d.\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] Current total training samples=%d, Unfreeze_batch_size=%d, training %d epochs, total training steps=%d.\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] Since total steps=%d < recommended %d, consider setting epochs to %d.\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   Backbone features are generic; freeze training can
    #   speed up training and protect weights early on.
    #   Init_Epoch     Start epoch
    #   Freeze_Epoch   Freeze training epochs
    #   UnFreeze_Epoch Total training epochs
    #   If OOM or out of memory, reduce batch_size.
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   Freeze part of the model
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   If not freezing, directly set batch_size to Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   Auto-adjust learning rate based on batch_size
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   Select optimizer by type
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   Get learning rate scheduler function
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        #---------------------------------------#
        #   Compute steps per epoch
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset too small for training. Please expand the dataset.")

        train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        #   Record eval mIoU curve
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        #---------------------------------------#
        #   Begin model training
        #---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            #---------------------------------------#
            #   If model has frozen layers,
            #   unfreeze and set parameters
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                #-------------------------------------------------------------------#
                #   Auto-adjust learning rate based on batch_size
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   Get learning rate scheduler function
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset too small for training. Please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler,
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=val_sampler,
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
