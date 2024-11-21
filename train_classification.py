import os
import sys
import torch
import numpy as np

import datetime
import provider
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.data_load import AtomicDataset
from train_utils import test, make_logger, create_dir

from models.point_nets.pointnet_cls import PointNet, point_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))




def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.006, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args = parse_args()
    exp_dir, checkpoints_dir, log_dir =  create_dir(args)
    logger = make_logger(args, log_dir)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    liquid_dataset = AtomicDataset(root="/home/teshbek/Work/PhD/PointCloudMaterials/datasets/Al/inherent_configurations_off",
                         data_files=["166ps.off"],
                         cube_size=16,
                         n_samples=20000,
                         num_point=256,
                         label=0)
    
    crystal_dataset = AtomicDataset(root="/home/teshbek/Work/PhD/PointCloudMaterials/datasets/Al/inherent_configurations_off",
                         data_files=["240ps.off"],
                         cube_size=16,
                         n_samples=20000,
                         num_point=256,
                         label=1)
    
    full_dataset = torch.utils.data.ConcatDataset([liquid_dataset, crystal_dataset])
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [int(0.8*len(full_dataset)), int(0.2*len(full_dataset))])
   
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=10, drop_last=True, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=10, shuffle=True)

    '''MODEL LOADING'''
    num_class = 2
    classifier = PointNet(num_class, normal_channel=args.use_normals)
    criterion = point_loss()

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

 
    start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        pbar = tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9)
        for batch_id, (points, target) in pbar:

            
            optimizer.zero_grad()

      
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            pbar.set_description("Loss %s" % round(loss.item(), 2))

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(args, classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
