import argparse
import time, itertools
import os, sys, datetime
import shutil, copy, math
from cityscapes import *
from collections import defaultdict

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
# sys.path.remove('/data1/chengtiancong/LLM_code/segment-anything-main')
sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from utils.distributed import *
from utils.logger import setup_logger
from utils.score import SegmentationMetric, DepthMetric, NormalMetric, RunningMetric
from utils.transformer import transformer, transformer_

from utils.flops import cal_multi_adds, cal_param_size
from models.model_zoo import get_segmentation_model
from losses import SegCrossEntropyLoss, depthloss, kdloss, make_criterion, train_transforms, valid_transforms, Weight, freeze_layer, loss_factory, model_fit, grad2vec, cagrad, overwrite_grad_segformer, grad2vec_segformer, DIST, nll


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='deeplabv3_multi_kd',
                        help='model name')  
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='backbone name')
    parser.add_argument('--task', type=str, default='multi',
                        help='type of task: single, multi, multi_mask')
    parser.add_argument('--dataset', type=str, default='city',
                        help='dataset name:citys,city,nyuv2')
    parser.add_argument('--data', type=str, default='/home/usr22100688/CV_Work/dataset/cityscapes-MTAN/',  #cityscapes256_512, NYU-V2, NYUD_InvPT
                        help='dataset directory')
    parser.add_argument('--crop-size', type=int, default=[512, 1024], nargs='+',
                        help='crop image size: [height, width]')
    parser.add_argument('--workers', '-j', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--flag', type=str, default='KD',
                        help='Type of KD: KD, soft_KD, grad_norm, progress, soft_aux_grad, grad_KD, amtl, pcgrad, com_mtl, bias_mtl')
    
    ## for amtl algorithm
    parser.add_argument('--loss_metric', default='amtl-a', help='methods for multi-task learning',
                        choices='amtl')
    parser.add_argument('--focusing_factor', default=2, type=float, help='Focusing factor for multi-task loss')
    parser.add_argument('--potential', nargs='+', default=[0.9683,0.9529], type=float,
                        help='Task potential of each task for achievement-based multi-task loss') #0.8152, 0.9865 for resnet-50, 0.8055,0.9852 for segformer B-0
    parser.add_argument('--margin', default=0.05, type=float, help='margin for task potential')
    parser.add_argument('--app', '--applications', nargs='+', default=['segmentation', 'depth'], help='detection, segmentation, depth')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help="Set zero if don't uses label smoothing")
    
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--val-batch', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--max-iterations', type=int, default=160000, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)') #for cityscapes:0.02, for nyuv2:
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--kd-weight', type=float, default=[1, 1, 1], nargs='+',
                        help='knowledge distillation weight: [weight1, 2, 3]')
    parser.add_argument('--loss-weight', type=float, default=[1.0, 1.0], nargs='+',
                        help='MTL loss weight: [weight1, 2, 3]')
    parser.add_argument('--sigma', type=float, default=10.0,
                        help='Variance of mask')
    parser.add_argument('--change-sigma', action='store_true', default=False,
                        help='Change variance of mask')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Hyperparameter in competition mtl')
    parser.add_argument('--ignore-label', type=int, default=-1, metavar='N',
                        help='input batch size for training (default: 8)')
    
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--device', type=str, default=None,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='./runs/checkpoint/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='./runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=400,
                        help='print log every log-iter')
    parser.add_argument('--alpha', type=float, default=1.5,
                        help='hyperparameters for grad_norm')
    parser.add_argument('--save-per-iters', type=int, default=800,
                        help='per iters to save')
    parser.add_argument('--val-per-iters', type=int, default=800,
                        help='per iters to val')
    parser.add_argument('--sig-semantic-iter', type=int, default=5000,
                        help='single semantic task iters')
    parser.add_argument('--sig-depth-iter', type=int, default=5000,
                        help='single depth task iters')
    parser.add_argument('--warm-up-iter', type=int, default=0,
                        help='MTL stu warm-up iters')
    parser.add_argument('--sig-total-iter', type=int, default=80000,
                        help='single tasks iters')
    parser.add_argument('--pretrained-base', type=str, default='resnet101-imagenet.pth', #'resnet18-imagenet.pth'
                        help='pretrained backbone')
    parser.add_argument('--pretrained', type=str, default='None',
                        help='pretrained seg model')
    parser.add_argument('--tea-pretrained', type=str, default='/home/usr22100688/CV_Work/code/MTL_work/MTL_citys/runs/checkpoint/teacher/',
                        help='pretrained teacher model')
    parser.add_argument('--write-com-value', action='store_true', default=False,
                        help='Change variance of mask')
    parser.add_argument('--unbalance', type=float, default=0.0,
                        help='Unbalance rate')

    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if num_gpus > 1 and args.local_rank == 0:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    # default settings for epochs, batch_size and lr

    # if args.backbone.startswith('resnet'):
    #     args.aux = True
    # elif args.backbone.startswith('mobile'):
    #     args.aux = False
    # else:
    #     raise ValueError('no such network')

    return args
    
def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        self.train_flag = 'multi'
        self.sig_iter = 1
        self.task_len = 2
        dataset_path = args.data

        if 'city' in args.dataset:
            train_dataset = Cityscapes(root=dataset_path, train=True, num_class=7, augmentation=True)
            val_dataset = Cityscapes(root=dataset_path, train=False)
        elif args.dataset == 'nyuv2':
            train_dataset = NYUD_MT(root=dataset_path, download=False, split='train', transform=train_transforms, do_edge=False, do_semseg=True, do_normals=True, do_depth= True, augmentation=False)
            val_dataset = NYUD_MT(root=dataset_path, download=False, split='val', transform=valid_transforms, do_edge=False, do_semseg=True, do_normals=True, do_depth= True)
            self.task_len = 3
        elif args.dataset == 'minist':
            train_dataset = MNIST(root=dataset_path, train=True, download=True, transform=global_transformer(), multi=True)
            val_dataset = MNIST(root=dataset_path, train=False, download=True, transform=global_transformer(), multi=True)
            self.task_len = 2
        else:
            raise ValueError('dataset unfind')

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model,
                                            backbone=args.backbone, 
                                            task_len=self.task_len,
                                            img_size=[256,512],
                                            pretrained=args.pretrained, 
                                            pretrained_base=args.pretrained_base,
                                            aux=args.aux, 
                                            norm_layer=BatchNorm2d,
                                            num_class=train_dataset.num_class).to(self.device)
        self.transformers = {}
        if 'segformer' in args.model:
            self.pretrained_layer_len = 188
            pretrain_path = args.tea_pretrained +  '/segformer_MiT_B1_city_256_{}.pth'
            self.tea1 = get_segmentation_model(model='segformer', backbone='MiT_B1', pretrained=pretrain_path.format('semantic'), pretrained_base='None', img_size=[256,512], norm_layer=BatchNorm2d, num_class=train_dataset.num_class).to(self.device)
            self.tea2 = get_segmentation_model(model='segformer', backbone='MiT_B1', pretrained=pretrain_path.format('depth'), pretrained_base='None', img_size=[256,512], norm_layer=BatchNorm2d, num_class=1).to(self.device)   
            # if args.flag == 'KDAM':
            pretrain_path = args.tea_pretrained + 'segformer_multi_MiT_B1_city_256.pth'
            # self.mul_tea = get_segmentation_model(model='segformer_multi', backbone='MiT_B1', pretrained=pretrain_path, pretrained_base='None', img_size=[256,512], norm_layer=BatchNorm2d, num_class=train_dataset.num_class).to(self.device)
            self.transformers = {}
            self.transformers[0]=transformer(256,256).to(self.device)
            self.transformers[1]=transformer(256,256).to(self.device)             
        elif 'deeplab' in args.model:
            pretrain_path = args.tea_pretrained + 'deeplabv3_resnet101_city_256_{}.pth'
            self.tea1 = get_segmentation_model(model='deeplabv3', backbone='resnet101', pretrained=pretrain_path.format('semantic'), pretrained_base='None', aux=True, norm_layer=BatchNorm2d, num_class=7).to(self.device)
            self.tea2 = get_segmentation_model(model='deeplabv3', backbone='resnet101', pretrained=pretrain_path.format('depth'), pretrained_base='None', aux=True, norm_layer=BatchNorm2d, num_class=1).to(self.device)
            
            if args.backbone == 'resnet18':
                self.transformers[0]=transformer(512,2048).to(self.device)
                self.transformers[1]=transformer(512,2048).to(self.device)
                self.transformers[0]=transformer(512,512).to(self.device)
                self.transformers[1]=transformer(512,512).to(self.device)
            elif args.backbone == 'resnet101':
                self.transformers[0]=transformer(2048,2048).to(self.device)
                self.transformers[1]=transformer(2048,2048).to(self.device)
            else:
                self.pretrained_layer_len = 159
                self.transformers[0]=transformer(2048,2048).to(self.device)
                self.transformers[1]=transformer(2048,2048).to(self.device)            
        else:
            self.pretrained_layer_len = 6
            pretrain_path = args.tea_pretrained + 'lenet_lenet_minist_{}.pth'
            self.tea1 = get_segmentation_model(model='lenet', backbone='lenet', pretrained=pretrain_path.format('semantic'), pretrained_base='None', aux=False, norm_layer=BatchNorm2d, num_class=10).to(self.device)
            self.tea2 = get_segmentation_model(model='lenet', backbone='lenet', pretrained=pretrain_path.format('depth'), pretrained_base='None', aux=False, norm_layer=BatchNorm2d, num_class=10).to(self.device)
            self.mul_tea = get_segmentation_model(model='lenet_multi', backbone='lenet', pretrained=args.tea_pretrained + 'lenet_multi_lenet_minist_single.pth', pretrained_base='None', aux=False, norm_layer=BatchNorm2d, num_class=10).to(self.device)
            self.transformers[0]=transformer(50,50).to(self.device)
            self.transformers[1]=transformer(50,50).to(self.device)

        with torch.no_grad():
            logger.info('Params: %.2fM'
                # % (cal_param_size(self.model) / 1e6, cal_multi_adds(self.model, (4, 3, 288, 384), device=self.device)/(1e9*2)))
                % (cal_param_size(self.model) / 1e6))
        
        args.batch_size = args.batch_size // num_gpus
        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iterations)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, args.val_batch)

        self.train_loader = data.DataLoader(dataset=train_dataset, 
                                            batch_sampler=train_batch_sampler,
                                            num_workers=args.workers,
                                            pin_memory=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)
        

        if  'multi' in args.task:
            if args.dataset != 'minist':
                self.criterion = SegCrossEntropyLoss(ignore_index=args.ignore_label).to(self.device)
                self.criterion1 = depthloss(args.ignore_label, self.device).to(self.device)
            else:
                self.criterion = nll()
            if 'soft_kd' in args.flag:
                self.criterion_kd = make_criterion()
            elif (('KD' in args.flag) and (args.flag != 'KDAM')):
                self.criterion_kd =  kdloss(weight=args.kd_weight)
            elif 'progress' in args.flag:
                self.criterion_kd = kdloss(weight=args.kd_weight)
                if 'segformer' in args.model:
                    self.layer_freeze = freeze_layer([w.clone().detach() for name,w in self.model.named_parameters() if w.requires_grad and not any(excluded in name for excluded in ['linear_fuse', 'linear_pred'])], ratio=0.8, end_ratio=0.2, max_iter=args.sig_total_iter)
                else:
                    self.layer_freeze = freeze_layer(self.model.pretrained.parameters(), ratio=0.8, end_ratio=0.2, max_iter=args.sig_total_iter)
            elif 'amtl' in args.flag:
                self.criterion_amtl = loss_factory(args, self.device)
            elif 'com_mtl' in args.flag:
                self.criterion_kd = make_criterion()
                # self.criterion_kd =  kdloss(weight=args.kd_weight)
                # self.criterion_kd = DIST()
            elif args.flag == 'KDAM':
                self.criterion_kd = DIST()

        # optimizer, for model just includes pretrained, head and auxlayer
        # params_list = []
        # params_list += self.model.parameters()
        params_list = [param for name,param in self.model.named_parameters() if 'clip' not in name]
        self.optimizer = torch.optim.Adam(params_list, lr=args.lr, weight_decay=args.weight_decay)

        if args.flag=='grad_norm':
            params_list = []
            self.Weights = Weight(task_len=self.task_len).to(self.device)
            params_list+=[self.Weights.weights]
            self.grad_optimizer = torch.optim.SGD(params_list, lr=0.001, momentum=0.1) #0.001 for resnet-50, 0.00001
            # self.grad_optimizer = torch.optim.Adam(params_list, lr=0.0001)
        if 'soft' not in args.flag:
            trans_params_list = []
            for i in range(len(self.transformers)):
                trans_params_list += self.transformers[i].parameters()
            self.trans_optimizer = torch.optim.Adam(trans_params_list, lr=1e-1)
            print(cal_param_size(self.transformers[0])*3/1e6)
        if ('cagrad' in args.flag) or ('com_mtl' in args.flag):
            self.grad_dims = []
            if 'segformer' in self.args.model: 
                for mm in [p for name,p in self.model.named_parameters() if p.requires_grad and not any(excluded in name for excluded in ['linear_pred'])]:
                    self.grad_dims.append(mm.data.numel())            
            elif 'deeplab' in self.args.model:   
                for mm in self.model.pretrained.parameters():
                    self.grad_dims.append(mm.data.numel())
            else:
                for mm in [p for name,p in self.model.named_parameters() if p.requires_grad and not any(excluded in name for excluded in ['task'])]:
                    self.grad_dims.append(mm.data.numel())                   
            self.grads = torch.Tensor(sum(self.grad_dims), 2).to(self.device)
        if 'clip_text' in args.backbone:
            params_list = [param for name,param in self.model.named_parameters() if ('clip' in name) and (param.requires_grad==True)]
            self.trans_optimizer = torch.optim.SGD(params_list, lr=0.002, momentum=0.9)
        # evaluation metrics
        if 'multi' in args.task:
            if args.dataset != 'minist':
                self.metric = SegmentationMetric(train_dataset.num_class)
                self.metric1 = DepthMetric(train_dataset.num_class, self.device)
            else:
                self.metric = RunningMetric(metric_type = 'ACC')
                self.metric1 = RunningMetric(metric_type = 'ACC')

        self.best_pred = 0.0
        self.best_err = 100.0

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False    
    def free(self, model):
        for param in model.parameters():
            param.requires_grad = True    
    def adjust_lr(self, base_lr, iter, max_iter, power):
        cur_lr = base_lr*((1-float(iter)/max_iter)**(power))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr

        return cur_lr

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def reduce_mean_tensor(self, tensor):
        rt = tensor.clone()
        # dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.num_gpus
        return rt

        
    def train(self):
        save_to_disk = get_rank() == 0
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_per_iters
        save_per_iters = self.args.save_per_iters
        start_time = time.time()
        logger.info('Start training, Total Iterations {:d}'.format(args.max_iterations))

        self.model.train()

        pre_param_list, cos_similarity,mask = [],0,0
        com_value,com_value_temp,num_share = [[0]*self.pretrained_layer_len for _ in range(self.task_len)],[[0]*self.pretrained_layer_len for _ in range(self.task_len)],0 
        if args.write_com_value:
            com_value_record = [[[] for _ in range(self.pretrained_layer_len)] for _ in range(self.task_len)]
            

        for iteration,  (images, label, depth) in enumerate(self.train_loader):
            kd_loss, semantic_loss, depth_loss, com_loss, com_I, record = 0,0,0,0,[[0]*self.pretrained_layer_len for _ in range(self.task_len)], 0
            iteration = iteration + 1
            # if args.warm_up_iter is not None:
            #     if iteration > args.warm_up_iter:
            #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=args.momentum, weight_decay=args.weight_decay)
            images = images.to(self.device) #torch.Size([4, 3, 288, 384]) 
            if 'multi' in args.task:
                depth = depth.to(self.device)
                label = label.long().to(self.device)
            # print(depth.shape, label.shape) #torch.Size([2, 1, 128, 256]) torch.Size([2, 128, 256])   

            if (args.unbalance!=0) & (iteration==1):
                p = args.unbalance
                mask = torch.bernoulli(torch.full((depth.size(0),), 1-p)).to(self.device) # output 1 in a prob:1-p
                # mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                # depth = depth*mask
                mask = mask.unsqueeze(1).unsqueeze(2)

            if 'progress' not in args.flag:
                outputs =  self.model(images)
                if args.flag != 'amtl':
                    if args.aux:
                        semantic_loss = model_fit(outputs[0][0], label, 'semantic') + 0.4 * model_fit(outputs[0][1], label, 'semantic')
                        depth_loss = model_fit(outputs[1][0], depth, 'depth') + 0.4 * model_fit(outputs[1][1], depth, 'depth')
                        # semantic_loss = semantic_loss*0.5
                        # depth_loss = depth_loss*1.5
                    else:
                        if args.unbalance:
                            label = label*mask.long()
                            non_zero_mask = mask.view(-1) > 0
                            depth_loss = model_fit(outputs[1][0], depth, 'depth')
                            for iter_de in range(len(label)):
                                if non_zero_mask[iter_de] == 0:
                                    semantic_loss+=0
                                else:
                                    semantic_loss += model_fit(outputs[0][0][iter_de,:].unsqueeze(0), label[iter_de,:].unsqueeze(0), 'semantic')
                            semantic_loss = semantic_loss/images.size(0)
                        else:
                            if args.dataset != 'minist':
                                semantic_loss = model_fit(outputs[0][0], label, 'semantic')
                                depth_loss = model_fit(outputs[1][0], depth, 'depth')
                            else:
                                semantic_loss, depth_loss = self.criterion(outputs[0], label), self.criterion(outputs[1], depth)
                    task_loss = semantic_loss+ depth_loss


                if args.flag=='grad_norm' or args.flag=='pcgrad':
                    norms = []
                    train_loss = []
                    train_loss.append(semantic_loss)
                    train_loss.append(depth_loss)
                    w = torch.ones(2).float().to(self.device)
                    # compute gradient w.r.t. last shared conv layer's parameters
                    if 'segformer' in args.model:
                        W = self.model.linear_fuse[0].weight
                    elif 'deeplab' in args.model:
                        W = self.model.pretrained.layer4[1].conv2.weight
                    else:
                        W = self.model.fc.weight

                    if args.flag == 'grad_norm':
                        for i in range(self.task_len):
                            gygw = torch.autograd.grad(train_loss[i], W, retain_graph=True)
                            norms.append(torch.norm(torch.mul(self.Weights.weights[i], gygw[0])))
                        norms = torch.stack(norms)
                        task_loss = torch.stack(train_loss)
                        if iteration == 1:
                            initial_task_loss = task_loss.detach()
                        loss_ratio = task_loss.data.detach() / initial_task_loss.data
                        inverse_train_rate = loss_ratio / loss_ratio.mean()
                        mean_norm = norms.mean()
                        constant_term = mean_norm.data * (inverse_train_rate ** args.alpha)
                        grad_norm_loss = (norms - constant_term).abs().sum()
                        for i in range(self.task_len):
                            w[i] = self.Weights.weights[i].data
                        task_loss = sum(w[i].data * train_loss[i] for i in range(self.task_len))

                        self.grad_optimizer.zero_grad()
                        grad_norm_loss.backward()
                        self.grad_optimizer.step()

                        norm_const = self.task_len/ sum(self.Weights.weights.data)
                        self.Weights.weights.data =  self.Weights.weights.data * norm_const
                        if args.write_com_value:
                            gygw = defaultdict(dict)
                            pre_param_list = [p for p in self.model.pretrained.parameters() if p.requires_grad]
                            train_loss = []
                            train_loss.append(w[0].data * semantic_loss)
                            train_loss.append(w[1].data * depth_loss)
                            # kd_loss_.append(kd_loss_1)
                            # kd_loss_.append(kd_loss_2)
                            for i in range(len(train_loss)):
                                # only calculate gradients but not accumulate and change the '.grad' params
                                gygw[i] = torch.autograd.grad(train_loss[i], pre_param_list, retain_graph=True)

                                # method 1
                                com_I[i] = [torch.sum(torch.abs(g*p.data)) for g,p in zip(gygw[i],pre_param_list)]
                                com_I[i] = torch.stack(com_I[i])
                                # com_I[i] = com_I[i] / torch.sum(com_I[i])
                                # print(len(com_I[i])) #184 for Mit_b0
                                com_value[i] = [x+y for x,y in zip(com_value[i], com_I[i])]

                            if iteration % 371 == 0: # city train data: 2975/ batch: 8 = 371
                                for i in range(len(train_loss)):
                                    com_value[i] = torch.stack(com_value[i])
                                    logger.info('Sum of Com-value{}: {}'.format(i,torch.sum(com_value[i])))
                                    com_value[i] = com_value[i] / torch.sum(com_value[i])

                                    for j in range(self.pretrained_layer_len):
                                        record_value = com_value[i][j].item()
                                        record_value = "%.4f" % record_value
                                        if com_value_record[i][j] == None:
                                            com_value_record[i][j] = record_value
                                        else:
                                            com_value_record[i][j].append(record_value)
                                com_value = [[0]*self.pretrained_layer_len for _ in range(self.task_len)]
                    
                    
                    elif args.flag == 'pcgrad':
                        gygw = torch.autograd.grad(train_loss[0], W, retain_graph=True)
                        gygw_dist = torch.autograd.grad(train_loss[1], W, retain_graph=True)
                        
                        shape = gygw[0].shape
                        gygw = gygw[0].flatten()
                        gygw_dist = gygw_dist[0].flatten()
                        cos_simi_0 = torch.dot(gygw, gygw_dist) / (torch.norm(gygw) * torch.norm(gygw_dist))
                        cos_similarity += cos_simi_0

                elif 'com_mtl' in args.flag:
                    with torch.no_grad():
                        self.tea1_outputs = self.tea1(images)
                        self.tea1_output = self.tea1_outputs[0].detach()
                        self.tea2_outputs = self.tea2(images)
                        self.tea2_output = self.tea2_outputs[0].detach()
                    kd_loss_1 = self.criterion_kd(outputs[0][0],self.tea1_output)
                    kd_loss_2 = self.criterion_kd(outputs[1][0],self.tea2_output)
                    ################################# KD2 #################################
                    kd_loss = kd_loss_1 + kd_loss_2
                    gygw = defaultdict(dict)
                    # com_curri_tem = 0.01 if iteration/args.max_iterations<0.01 else iteration/args.max_iterations*0.5

                    if iteration > self.args.warm_up_iter:
                        if 'segformer' in self.args.model:
                            pre_param_list = [p.contiguous() for name,p in self.model.named_parameters() if p.requires_grad and not any(excluded in name for excluded in ['linear_pred'])]
                        elif 'deeplab' in self.args.model:
                            # if (args.flag=='com_mtl_t') & ((iteration-1)%371 == 0):
                            #     w_init = [w.clone().detach() for w in self.model.pretrained.parameters() if w.requires_grad]
                            pre_param_list = [p for p in self.model.pretrained.parameters() if p.requires_grad]
                        else:
                            pre_param_list = [p.contiguous() for name,p in self.model.named_parameters() if p.requires_grad and not any(excluded in name for excluded in ['task'])]
                            
                        layer_index = int(math.ceil((1 - iteration/args.max_iterations) * self.pretrained_layer_len))
                        pre_param_list = pre_param_list[layer_index:self.pretrained_layer_len]

                        train_loss = []
                        train_loss.append(semantic_loss+kd_loss_1)
                        train_loss.append(depth_loss+kd_loss_2)
                        
                        # if imp_cal:
                        #     cnt += 1
                        if len(pre_param_list) != 0:
                            for i in range(len(train_loss)):
                                gygw[i] = torch.autograd.grad(train_loss[i], pre_param_list, retain_graph=True)

                                # com_I[i] = [torch.sum(torch.abs(g*p.data)) for g,p in zip(gygw[i],pre_param_list)]
                                # com_I[i] = torch.stack(com_I[i])
                                # com_I[i] = (com_I[i] / torch.sum(com_I[i]))+1e-6
                                # # print(len(com_I[i])) #184 for Mit_b0
                                # com_value_temp[i] = [x+y for x,y in zip(com_value[i], com_I[i])]

                                # method 1
                                com_I[i][layer_index:self.pretrained_layer_len] = [torch.sum(torch.abs(g*p.data)) for g,p in zip(gygw[i],pre_param_list)]
                                com_value = torch.stack(com_I[i][layer_index:self.pretrained_layer_len])
                                com_I[i][layer_index:self.pretrained_layer_len] = com_value / (torch.sum(com_value)+1e-6)

                elif args.flag == 'KDAM':
                    with torch.no_grad():
                        self.tea_outputs = self.mul_tea(images)
                        self.tea1_output = self.tea_outputs[0].detach()
                        self.tea2_output = self.tea_outputs[1].detach()

                    kd_loss = self.criterion_kd(outputs[0],self.tea1_output) + self.criterion_kd(outputs[1],self.tea2_output)

                elif (('KD' in args.flag) and (args.flag != 'KDAM')):
                    with torch.no_grad():
                        self.tea1_outputs = self.tea1(images)
                        self.tea1_output = self.tea1_outputs[-1].detach() #torch.Size([8, 256, 64, 128]) for mit_b1
                        self.tea2_outputs = self.tea2(images)
                        self.tea2_output = self.tea2_outputs[-1].detach()
                    self.stu1_output = self.transformers[0](outputs[-1])
                    self.stu2_output = self.transformers[1](outputs[-1])
                    kd_loss_1,_ = self.criterion_kd(iteration-1, s1_output=self.stu1_output, t1_output=self.tea1_output)
                    kd_loss_2,_ = self.criterion_kd(iteration-1, s2_output=self.stu2_output, t2_output=self.tea2_output)
                    kd_loss = kd_loss_1 + kd_loss_2

                    if args.flag == 'grad_KD':
                        W = self.model.pretrained.layer4[1].conv2.weight
                        gygw = torch.autograd.grad(task_loss, W, retain_graph=True)
                        gygw_dist = torch.autograd.grad(kd_loss, W, retain_graph=True)       

                elif 'soft_kd' in args.flag:
                    with torch.no_grad():
                        self.tea1_outputs = self.tea1(images)
                        self.tea1_output = self.tea1_outputs[0].detach()
                        self.tea2_outputs = self.tea2(images)
                        self.tea2_output = self.tea2_outputs[0].detach()

                        if 'lenet' not in args.model:
                            kd_loss_1 = self.criterion_kd(outputs[0][0],self.tea1_output)
                            kd_loss_2 =  self.criterion_kd(outputs[1][0],self.tea2_output)
                        else:
                            kd_loss_1, kd_loss_2 = self.criterion_kd(outputs[0],self.tea1_output), self.criterion_kd(outputs[1],self.tea2_output)
                        kd_loss=kd_loss_1+kd_loss_2                
                
                elif args.flag == 'amtl':
                    with torch.no_grad():
                        self.tea1_outputs = self.tea1(images)
                        self.tea1_output = self.tea1_outputs[0].detach()
                        self.tea2_outputs = self.tea2(images)
                        self.tea2_output = self.tea2_outputs[0].detach()
                    if args.aux:
                        task_loss = self.criterion_amtl({'segmentation': outputs[0][0], 'depth': outputs[1][0]}, gt={'segmentation': label, 'depth': depth}, pseudo={'segmentation': self.tea1_output, 'depth': self.tea2_output}) + 0.4 * (model_fit(outputs[0][1], label, 'semantic') + model_fit(outputs[1][1], depth, 'depth'))
                    else:
                        task_loss = self.criterion_amtl({'segmentation': outputs[0][0], 'depth': outputs[1][0]}, gt={'segmentation': label, 'depth': depth}, pseudo={'segmentation': self.tea1_output, 'depth': self.tea2_output})



            ################################################## Loss Backward ################################################################
            losses = task_loss + kd_loss

            lr = self.adjust_lr(base_lr=args.lr, iter=iteration-1, max_iter=args.max_iterations, power=0.9)
            # if args.warm_up_iter is not None:
            #     if iteration > args.warm_up_iter:
            #         lr = self.adjust_lr(base_lr=0.0001, iter=iteration-1, max_iter=args.max_iterations, power=0.9)
            self.optimizer.zero_grad()
            if ((kd_loss != 0) & ('soft' not in args.flag)) or ('clip_text' in args.backbone):
                self.trans_optimizer.zero_grad()


            if 'grad_optim' not in args.flag:
                losses.backward()

            # print(self.model.clip_model.model_1.prompt_learner.ctx)


            if args.flag == 'grad_KD':
                shape = gygw[0].shape
                gygw = gygw[0].flatten()
                gygw_dist = gygw_dist[0].flatten()
                cos_simi_0 = torch.dot(gygw, gygw_dist) / (torch.norm(gygw) * torch.norm(gygw_dist)) 
                if cos_simi_0 <0:                 
                    gygw_dist -= torch.dot(gygw, gygw_dist) * gygw / (gygw.norm()**2)
                    self.model.pretrained.layer4[1].conv2.weight.grad = gygw.view(shape) + gygw_dist.view(shape)  
            elif self.args.flag=='pcgrad':
                if cos_simi_0 < 0:
                    gygw_dist -= torch.dot(gygw, gygw_dist) * gygw / (gygw.norm()**2)
                    if 'segformer' in args.model:
                        self.model.linear_fuse[0].weight.grad = gygw.view(shape)+gygw_dist.view(shape)  
                    elif 'deeplab' in args.model:
                        self.model.pretrained.layer4[1].conv2.weight.grad = gygw.view(shape)+gygw_dist.view(shape)  
                    else:
                        self.model.fc.weight.grad = gygw.view(shape)+gygw_dist.view(shape)  
                    cos_similarity -= cos_simi_0
            elif 'com_mtl' in self.args.flag:

                    w_a = 0
                    if (iteration > self.args.warm_up_iter) & (len(pre_param_list) != 0):
                        assert len(com_I[0]) == self.pretrained_layer_len
                        for num, (param, a,b) in enumerate(zip(pre_param_list, com_I[0], com_I[1])):
                            # if num >= self.pretrained_layer_len * 0.8:
                            #     param.grad = gygw[1][num]
                            # enhance good gradient weight
                            if num >= (1 - iteration/args.max_iterations) * self.pretrained_layer_len:
                            # if num > (iteration/args.max_iterations) * self.pretrained_layer_len:
                                # reduce bad gradient weight
                                w_a = 1 if a>=b else (a/b)**self.args.temp
                                w_b = 1 if b>=a else (b/a)**self.args.temp
                                param.grad = w_a*gygw[0][num] + w_b*gygw[1][num]
                                # enlarge good gradient weight
                                # w_a = (a/b)**self.args.temp if a>=b else 1
                                # w_b = (b/a)**self.args.temp if b>=a else 1
                            # if num < (1 - iteration/args.max_iterations) * self.pretrained_layer_len:
                            #     w_a = 1 if a>=b else (a/b)**self.args.temp
                            #     w_b = 1 if b>=a else (b/a)**self.args.temp

                            # w_a = 0 if w_a<1e-1 else w_a
                            # w_b = 0 if w_b<1e-1 else w_b
                            # if w_a + w_b == 1: #one grad is none
                            #     record+=1
                            
            elif 'grad_optim' in args.flag:
                if args.flag == 'cagrad_optim':
                    for i in range(2):
                        if i == 0:
                            semantic_loss.backward(retain_graph=True)
                        else:
                            depth_loss.backward()
                        grad2vec_segformer(self.model, self.grads, self.grad_dims, i)
                        self.model.zero_grad_shared_modules()
                    g = cagrad(self.grads, args.alpha, rescale=1)
                    overwrite_grad_segformer(self.model, g, self.grad_dims, self.device)

            
            self.optimizer.step()
            if ((kd_loss != 0) & ('soft' not in args.flag)) or ('clip_text' in args.backbone):
                self.trans_optimizer.step()


            # reduce losses over all GPUs for logging purposes
            task_losses_reduced = self.reduce_mean_tensor(task_loss)
            if not isinstance(kd_loss, int):
                kd_loss = kd_loss.item()
            if not isinstance(semantic_loss, int):
                semantic_loss = semantic_loss.item()
            if not isinstance(depth_loss, int):
                depth_loss = depth_loss.item()
            if not isinstance(com_loss, int):
                com_loss = com_loss.item()
            eta_seconds = ((time.time() - start_time) / iteration) * (args.max_iterations - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and save_to_disk:
                if self.args.flag == 'com_mtl': #[2,layer_len,100000/371]
                    logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || Kd Loss: {:.4f} || Semantic Loss: {:.4f} || Depth Loss: {:.4f} || Competition: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'], task_losses_reduced.item(), kd_loss, semantic_loss, depth_loss, record,
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))
                else:
                    logger.info(
                    "Iters: {:d}/{:d} || Lr: {:.6f} || Task Loss: {:.4f} || Kd Loss: {:.4f} || Semantic Loss: {:.4f} || Depth Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        iteration, args.max_iterations, self.optimizer.param_groups[0]['lr'], task_losses_reduced.item(), kd_loss, semantic_loss, depth_loss,
                        str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

            if iteration % save_per_iters == 0 and save_to_disk:
                save_checkpoint(self.model, self.args, is_best=False)

            if not self.args.skip_val and iteration % val_per_iters == 0:
                self.validation()
                self.model.train()
                if args.flag == 'grad_norm':
                    logger.info("Weight of each task: {} and {}".format(self.Weights.weights[0].data, self.Weights.weights[1].data))
                elif args.flag == 'pcgrad':
                    logger.info("Cos similarity: {}".format(cos_similarity))
                elif args.flag == 'com_mtl':
                    logger.info('Params_lenth: {} and W_a: {}'.format(len(pre_param_list), w_a))

        save_checkpoint(self.model, self.args, is_best=False)
        if args.write_com_value:
            if args.flag == 'com_mtl':
                with open('./runs/logs/competition_value_{}_{}_{}.txt'.format(args.dataset, args.backbone, args.flag), 'w') as file:
                    # record_value = copy.deepcopy(com_value[i])
                    # record_value = record_value.detach().cpu()
                    # record_value = ['{:.4f}'.format(num) for num in record_value]
                    file.write("Competition Value: {}\n".format(com_value_record))
            else:
                with open('./runs/logs/competition_value_{}_{}_{}.txt'.format(args.dataset, args.backbone,args.flag), 'w') as file:
                    file.write("Competition Value: {}\n".format(com_value_record))
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / args.max_iterations))


    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        self.metric1.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        
        for i, (image, label, depth)  in enumerate(self.val_loader):
            image = image.to(self.device)
            if 'multi' in args.task:
                label = label.long().to(self.device)
                depth = depth.to(self.device)

            with torch.no_grad():
                outputs = model(image)

            if args.dataset != 'minist':
                B, C, H, W = depth.size()
                outputs[0][0] = F.interpolate(outputs[0][0], (H, W), mode='bilinear', align_corners=True)
                outputs[1][0] = F.interpolate(outputs[1][0], (H, W), mode='bilinear', align_corners=True)

                self.metric.update(outputs[0][0], label)
                self.metric1.update(outputs[1][0], depth)
            else:
                self.metric.update(outputs[0], label)
                self.metric1.update(outputs[1], depth)

            if 'multi' in args.task:
                pixAcc, mIoU = self.metric.get()
                abs_err, rel_err = self.metric1.get()

        if self.args.flag == 'amtl':
            if hasattr(self.criterion_amtl, 'update_kpi'):
                self.criterion_amtl.update_kpi({'mIoU': mIoU, 'd1': 1-abs_err})

        if 'multi' in args.task:
            logger.info("Sample: {:d}, Validation pixAcc: {:.4f}, mIoU: {:.4f}, Abs_error: {:.4f}, Rel_error: {:.4f}"
                                .format(i + 1, pixAcc, mIoU, abs_err, rel_err))

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        new_err = rel_err
        if new_err < self.best_err:
            self.best_err = new_err

        if args.local_rank == 0:
            save_checkpoint(self.model, self.args, is_best)
        synchronize()

def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset, args.flag)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_model.pth'.format(args.model, args.backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)

if __name__ == '__main__':
    args = parse_args()
    seed_ = random.randint(0,10000)
    seed_ = random.choice([9285, 7767, 4468, 578, 4025, 8793, 8540,8471,8677,8832,3961,7937])
    seed_ = 8667
    fix_random_seed(seed_)
    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = False
        if args.device is None:
            args.device = "cuda:5"
    else:
        args.distributed = False
        if args.device is None:
            args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    logger = setup_logger("multi", args.log_dir, get_rank(), filename='{}_{}_{}_{}_log.txt'.format(
        args.model, args.backbone, args.dataset, args.flag))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    logger.info('random seed: {}'.format(seed_))

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
