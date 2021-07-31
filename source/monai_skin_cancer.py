# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import json
import logging
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchtext
import numpy as np
from PIL import Image
from monai.config import print_config
from monai.transforms import \
    Compose, LoadPNG, Resize, AsChannelFirst, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom
from monai.networks.nets import densenet121
from monai.metrics import compute_roc_auc
from skin_cancer_dataset import SkinCancerDataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _get_train_data_loader(batch_size, trainX, trainY, is_distributed, **kwargs):
    logger.info("Get train data loader")
    
    train_transforms = Compose([
        LoadPNG(image_only=True),
        AsChannelFirst(channel_dim=2),
        ScaleIntensity(),
        RandRotate(range_x=15, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        Resize(spatial_size=(64,64)),
        ToTensor()
    ])
    
    dataset = SkinCancerDataset(trainX, trainY, train_transforms)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, **kwargs)


def _get_test_data_loader(batch_size, valX, valY, **kwargs):
    
    val_transforms = Compose([
        LoadPNG(image_only=True),
        AsChannelFirst(channel_dim=2),
        ScaleIntensity(),
        Resize(spatial_size=(64,64)),
        ToTensor()
    ])
    
    dataset = SkinCancerDataset(valX, valY, val_transforms)
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(args):
    
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.debug('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    
    torchtext.utils.extract_archive(args.data_dir+'/HAM10000.tar.gz', args.data_dir)
    
    #build file lists
    data_dir = args.data_dir+'/HAM10000/train_dir'
    class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
    num_class = len(class_names)
    image_files = [[os.path.join(data_dir, class_name, x) 
            for x in os.listdir(os.path.join(data_dir, class_name))] 
            for class_name in class_names]
    image_file_list = []
    image_label_list = []

    for i, class_name in enumerate(class_names):
        image_file_list.extend(image_files[i])
        image_label_list.extend([i] * len(image_files[i]))
    num_total = len(image_label_list)
    image_width, image_height = Image.open(image_file_list[0]).size

    valid_frac, test_frac = 0.1, 0.1
    trainX, trainY = [], []
    valX, valY = [], []
    testX, testY = [], []

    for i in range(num_total):
        rann = np.random.random()
        if rann < valid_frac:
            valX.append(image_file_list[i])
            valY.append(image_label_list[i])
        elif rann < test_frac + valid_frac:
            testX.append(image_file_list[i])
            testY.append(image_label_list[i])
        else:
            trainX.append(image_file_list[i])
            trainY.append(image_label_list[i])

    print("Training count =",len(trainX),"Validation count =", len(valX), "Test count =",len(testX))
            
    train_loader = _get_train_data_loader(args.batch_size, trainX, trainY, False, **kwargs)
    val_loader = _get_test_data_loader(args.test_batch_size, valX, valY, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, testX, testY, **kwargs)

    #create model
    model = densenet121(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_class
    ).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    epoch_num = args.epochs
    val_interval = 1
    
    #train model
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    for epoch in range(epoch_num):
        logger.info('-' * 10)
        logger.info(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            logger.info(f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
            epoch_len = len(train_loader.dataset) // train_loader.batch_size        
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
                metric_values.append(auc_metric)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if auc_metric > best_metric:
                    best_metric = auc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), 'best_metric_model.pth')
                    logger.info('saved new best metric model')
                logger.info(f"current epoch: {epoch + 1} current AUC: {auc_metric:.4f}"
                      f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f}"
                      f" at epoch: {best_metric_epoch}")
    logger.info(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")        
   
    save_model(model, args.model_dir) 
    
    #test data:classification report
    model.load_state_dict(torch.load('best_metric_model.pth'))
    model.to(device)
    model.eval()
    y_true = list()
    y_pred = list()
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    from sklearn.metrics import classification_report
    logger.info(classification_report(y_true, y_pred, target_names=class_names, digits=4))
            
            
def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = densenet121(
        spatial_dims=2,
        in_channels=3,
        out_channels=7
    )    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)   
    
def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    
    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())
