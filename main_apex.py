from ops.dataset import TSNDataSet
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool
from torchsummary import summary
from tensorboardX import SummaryWriter
from src.label_smoothing import LabelSmoothingCrossEntropy
from apex import amp
from apex.parallel import DistributedDataParallel
import apex

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch APEX Training')

parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff', 'HP', 'FlowNet', 'TVNet'], default='HP')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32, the total bach size of a node)')
parser.add_argument('--batch_multiplier', default=1, type=int, metavar='N')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N')
parser.add_argument('--dataset', type=str)
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--lr_type', default='step', type=str, metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--root-model', default='checkpoint', type=str)
parser.add_argument('--clip-gradient', '--gd', default=None, type=float, metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")
parser.add_argument('--prefix', type=str, default='FC')
parser.add_argument('--suffix', type=str, default='1')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrain', default='imagenet', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://ip_address:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--available_gpus', default='0, 1, 2, 3', type=str)
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--eval-freq', '-ef', default=1, type=int, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N', help='number of data loading workers (default: 30)')
parser.add_argument('--sbn', default=False, action="store_true", help='enable SyncBatchNorm')
parser.add_argument('--ls', default=False, action="store_true", help='enable LabelSmoothing')
parser.add_argument('--arch', type=str, default="resnet50")
# parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')

best_acc1 = 0
global global_time
global_time = time.time()


def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.available_gpus
    args.loss_type ='nll'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    args.num_class, args.train_list, args.val_list, args.root_path, args.image_tmpl = dataset_config.return_dataset(args.dataset, args.modality)
    args.store_name = '_'.join([args.prefix, args.dataset, args.modality, args.arch, 'segment%d' % args.num_segments, 'e{}'.format(args.epochs)])
    if args.lr_type == 'step':
        _steps = np.array(args.lr_steps).astype('int32')          
        args.store_name += '_{}{}'.format(args.lr_type, _steps)
    else:
        args.store_name += '_{}'.format(args.lr_type)
    if args.dense_sample:
        args.store_name += '_dense'
#     if args.non_local > 0:
#         args.store_name += '_nl'
    args.store_name += '_lr{}'.format(args.lr)
    args.store_name += '_wd{:.1e}'.format(args.weight_decay)
    args.store_name += '_do{}'.format(args.dropout)
    if args.warmup_epochs > 0:
        args.store_name += '_wup{}'.format(args.warmup_epochs)
    args.store_name += '_sbz{}x{}x{}'.format(args.batch_size, args.world_size, args.batch_multiplier)
    if args.sbn:
        args.store_name += '_sbn'
    if args.ls:
        args.store_name += '_ls'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)
    if args.rank == 0 and not args.evaluate:
        check_rootfolders(args)
    
    if args.modality == 'RGB':
        args.data_length = 1
    elif args.modality == 'HP':
        args.data_length = 3
    elif args.modality in ['Flow', 'RGBDiff']:
        args.data_length = 5
    elif args.modality  in ['FlowNet', 'TVNet']:
        args.data_length = 6
    
        
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        print('WORLD_SIZE={}'.format(args.world_size))
        print('NPGPUS_PER_NODE={}'.format(ngpus_per_node))
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("rank-{} is runing".format(args.rank))
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

#--------------------------------------------------------------------------------------------------------------------------

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=rank)
    else:
        rank = 0
        
    if args.prefix == 'FC':
        from src.model import TSN
    elif args.prefix == 'F':
        from src.model_busy import TSN
    else:
        from ops.models import TSN
    
    if args.modality == 'FlowNet':
        from FlowNet2.models import FlowNet2S
        args.fp16 = False
        args.rgb_max = 255
        args.flownet = FlowNet2S(args)
        dict = torch.load("FlowNet2/weights/FlowNet2-S_checkpoint.pth.tar")
        args.flownet.load_state_dict(dict["state_dict"])
        if args.distributed:
            if args.gpu is not None:
                args.flownet.cuda(args.gpu)
            else:
                args.flownet.cuda()
        elif args.gpu is not None:
            args.flownet.cuda(args.gpu)
    if args.modality == 'TVNet':
        from TVNet.tvnet import TVNet
        if args.distributed:
            if args.gpu is not None:
                args.tvnet = TVNet(device=torch.device("cuda:{}".format(args.gpu))).cuda(args.gpu)
            else:
                args.tvnet = TVNet(device=torch.device("cuda")).cuda()
        elif args.gpu is not None:
            args.tvnet = TVNet(device=torch.device("cuda:{}".format(args.gpu))).cuda(args.gpu)
    # create model
    if 'F' in args.prefix:   
        model = TSN(args.num_class, args.num_segments, args.data_length, args.arch,
                    dropout=args.dropout, partial_bn=not args.no_partialbn, print_spec=True, fc_lr5=True, modality=args.modality)
    else:
        model = TSN(args.num_class, args.num_segments, args.modality,
                    base_model=args.arch,
                    consensus_type='avg',
                    dropout=args.dropout,
                    partial_bn=not args.no_partialbn,
                    pretrain=args.pretrain,
                    is_shift=True, shift_div=8, shift_place='blockres',
                    fc_lr5=True,
                    temporal_pool=False,
                    non_local=False)
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        else:
            model.cuda()
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    policies = model.get_optim_policies()
    optimizer = optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    if rank == 0:
        print(model)
    args.crop_size = model.crop_size
    args.scale_size = model.scale_size
    args.input_mean = model.input_mean
    args.input_std = model.input_std
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    if args.distributed:
        if args.gpu is not None:
            model = DistributedDataParallel(model, delay_allreduce=True)
            if args.sbn:
                model = apex.parallel.convert_syncbn_model(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, delay_allreduce=True)
            if args.sbn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
    elif args.gpu is not None:
        pass
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    # define loss function (criterion) and optimizer
    if args.ls:
        criterion = LabelSmoothingCrossEntropy(reduction='mean').cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    if args.pretrain and args.pretrain != 'imagenet':
        if os.path.isfile(args.pretrain):
            print("=> loading pretrained '{}'".format(args.pretrain))
            if args.gpu is None:
                checkpoint = torch.load(args.pretrain)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.pretrain, map_location=loc)
                pretrained_dict = checkpoint['state_dict']
                model_dict = model.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict)
                
            print("=> loaded pretrained '{}' ".format(args.pretrain))
            del checkpoint 
        else:
            print("=> no pretarined found at '{}'".format(args.pretrain))
            
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            if args.start_epoch == 1:
                args.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint 
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            

        

    cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(args.input_mean, args.input_std)
    else:
        normalize = IdentityTransform()
    train_process = [train_augmentation,
                     Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                     ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])), normalize]
    val_process = [GroupScale(int(args.scale_size)),
                   GroupCenterCrop(args.crop_size),
                   Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                   ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])), normalize]


    train_dataset = TSNDataSet(args.dataset, args.root_path, args.train_list, num_segments=args.num_segments,
                       data_length=args.data_length,
                       modality=args.modality,
                       image_tmpl=args.image_tmpl,
                       random_shift=True,
                       transform=torchvision.transforms.Compose(train_process), 
                      dense_sample=args.dense_sample)

    val_dataset = TSNDataSet(args.dataset, args.root_path, args.val_list, num_segments=args.num_segments, 
                             data_length=args.data_length,
                             modality=args.modality,
                             image_tmpl=args.image_tmpl,
                             random_shift=False,
                             transform=torchvision.transforms.Compose(val_process), 
                             dense_sample=args.dense_sample)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                                               num_workers=args.workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                             num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args, rank)
        return
    
    log = None
    if rank == 0:
        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    
        log = open(os.path.join(args.root_model, args.store_name, 'log.csv'), 'a')
        with open(os.path.join(args.root_model, args.store_name, 'args.txt'), 'w') as f:
            f.write(str(args))
            
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_model, args.store_name)) if rank == 0 else None
    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, log, tf_writer, args, rank)
        if rank == 0:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'amp': amp.state_dict(),
            }, False, args)
        
        if epoch % 10 == 0 and rank == 0:
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
            }, False, args, e=epoch)

        # evaluate on validation set
        is_best = False
        if (epoch % args.eval_freq == 0 or epoch == args.epochs):
            acc1 = validate(val_loader, model, criterion, args, rank, epoch, log, tf_writer)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
        if tf_writer is not None:
            tf_writer.flush()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and rank == 0):
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'amp': amp.state_dict(),
            }, is_best, args)
    if tf_writer is not None:
        tf_writer.close()
#--------------------------------------------------------------------------------------------------------------------------

def train(train_loader, model, criterion, optimizer, epoch, log, tf_writer, args, rank):
    
    torch.cuda.empty_cache()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    model.zero_grad()
    loss_tmp = []
    acc_tmp = []
    for i, (input, target) in enumerate(train_loader):
        if i % args.batch_multiplier == 0:
            adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps, args, (epoch-1) + float(i) / len(train_loader))
        i += 1
        # measure data loading time
        data_time.update(time.time() - end)
        
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        target_var = target
        if args.modality == 'FlowNet':
            args.flownet.eval()
            with torch.no_grad():
                #n*t*data_length, c, h, w
                input = input.detach()
                input = F.interpolate(input, size=[256, 256], mode='bilinear', align_corners=True, recompute_scale_factor=True)
                input = input.reshape(-1, args.data_length, 3, 256, 256).transpose(1,2) #nt, c, data_length, h, w
                input_list = []
                for _l in range(args.data_length - 1):
                    input_list.append(args.flownet(input[:,:,_l:_l+2]))
                input = torch.stack(input_list, dim=1) #nt, data_length-1, c, h, w
                input = input.reshape(-1, 2, 256, 256)
                input = F.interpolate(input, size=[224, 224], mode='bilinear', align_corners=True, recompute_scale_factor=True)  #nt(data_length-1), 2, h, w
        if args.modality == 'TVNet':
            args.tvnet.eval()
            with torch.no_grad():
                input = input.detach()
                input = input.reshape(-1, args.data_length, 3, 224, 224) #nt, data_length, 3, h, w
                input = input.mean(dim=2, keepdim=True) #nt, data_length, 1, h, w
                input_list = []
                for _l in range(args.data_length - 1):
                    u1, u2, _ = args.tvnet(input[:,_l], input[:,_l+1])
                    flow = torch.stack((u1, u2), dim=1)
                    input_list.append(flow)
                input = torch.stack(input_list, dim=1) #nt, data_length-1, 2, h, w
                input = input.reshape(-1, 2, 224, 224)

        # compute output
        output = model(input)
        loss = criterion(output, target_var) / args.batch_multiplier


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item()*args.batch_multiplier, input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        loss_tmp.append(loss.item()*args.batch_multiplier)
        acc_tmp.append(prec1.item())

        # compute gradient and do SGD step
        if i % args.batch_multiplier != 0:
            with amp.scale_loss(loss, optimizer, delay_unscale=True) as scaled_loss:
                scaled_loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if args.clip_gradient is not None:
                clip_grad_norm_(amp.master_params(optimizer), args.clip_gradient)
                
            optimizer.step()
            
        if rank == 0:
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % (args.print_freq*args.batch_multiplier*5) == 0 and tf_writer is not None:
                tf_writer.add_scalar('loss/step', np.mean(loss_tmp), ((epoch-1)*len(train_loader)+i))
                tf_writer.add_scalar('acc/step', np.mean(acc_tmp), ((epoch-1)*len(train_loader)+i))
                loss_tmp = []
                acc_tmp = []
            if i % (args.print_freq * args.batch_multiplier) == 0:
                output = ('Epoch: [{0:3d}][{1:4d}/{2:4d}], lr: {lr:.5f}\t'
                          'Time {time:.1f}\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, int(i/args.batch_multiplier), int(len(train_loader)/args.batch_multiplier), time=(time.time()-global_time)/60.,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
                print(output)
                log.write(output + '\n')
                log.flush()
#                 if i % args.print_freq == 0 and tf_writer is not None:
#                     for tag, value in model.named_parameters():
#                         if (not value is None) and (not value.grad is None):
#                             tag = tag.replace('.', '/')
#                             tf_writer.add_histogram('weights/'+tag, value.detach(), (epoch*len(train_loader)+i)/args.batch_multiplier)
#                             tf_writer.add_histogram('grads/'+tag, value.grad.detach().abs().mean(), (epoch*len(train_loader)+i)/args.batch_multiplier)
                        
        if i % args.batch_multiplier == 0:
            optimizer.zero_grad()
            
    if tf_writer is not None:
        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        
    

def validate(val_loader, model, criterion, args, rank, epoch=None, log=None, tf_writer=None):
    
    def metric_average(val):
        tensor = torch.tensor(val).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        return tensor.item() / args.world_size
    #     avg_tensor = hvd.allreduce(tensor, name=name)
    #     return avg_tensor.item()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            i += 1
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            if args.modality == 'FlowNet':
                args.flownet.eval()
                with torch.no_grad():
                    #n*t*data_length, c, h, w
                    input = input.detach()
                    input = F.interpolate(input, size=[256, 256], mode='bilinear', align_corners=True, recompute_scale_factor=True)
                    input = input.reshape(-1, args.data_length, 3, 256, 256).transpose(1,2) #nt, c, data_length, h, w
                    input_list = []
                    for _l in range(args.data_length - 1):
                        input_list.append(args.flownet(input[:,:,_l:_l+2]))
                    input = torch.stack(input_list, dim=1) #nt, data_length-1, c, h, w
                    input = input.reshape(-1, 2, 256, 256)  #nt(data_length-1), 2, h, w
                    input = F.interpolate(input, size=[224, 224], mode='bilinear', align_corners=True, recompute_scale_factor=True)  #nt(data_length-1), 2, h, w
#                     input = input.div(255.)
#                     input.sub_(args.input_mean[0]).div_(args.input_std[0])
            if args.modality == 'TVNet':
                args.tvnet.eval()
                with torch.no_grad():
                    input = input.detach()
                    input = input.reshape(-1, args.data_length, 3, 224, 224) #nt, data_length, 3, h, w
                    input = input.mean(dim=2, keepdim=True) #nt, data_length, 1, h, w
                    input_list = []
                    for _l in range(args.data_length - 1):
                        u1, u2, _ = args.tvnet(input[:,_l], input[:,_l+1])
                        flow = torch.stack((u1, u2), dim=1)
                        input_list.append(flow)
                    input = torch.stack(input_list, dim=1) #nt, data_length-1, 2, h, w
                    input = input.reshape(-1, 2, 224, 224)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and rank == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
                if log is not None:
                    log.write(output + '\n')
                    log.flush()
                    
    top1.avg = metric_average(top1.avg)
    top5.avg = metric_average(top5.avg)
    losses.avg = metric_average(losses.avg)
    if rank == 0:
        output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(top1=top1, top5=top5, loss=losses))
        print(output)
        if log is not None:
            log.write(output + '\n')
            log.flush()
        
    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.flush()

    return top1.avg



def save_checkpoint(state, is_best, args, e=None):
    if e is not None:
        filename = '%s/%s/epoch%d_ckpt.pt' % (args.root_model, args.store_name, e)
    else:
        filename = '%s/%s/ckpt.pt' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pt', '.best.pt'))
       
        
def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps, args, epoch_float=-1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch_float < 0:
        epoch_float = float(epoch)
    if epoch_float < args.warmup_epochs:
        base_lr = args.lr / 10
        lr = (epoch_float / args.warmup_epochs) * (args.lr - base_lr) + base_lr
        decay = args.weight_decay
    elif lr_type == 'step':
        decay = 0.1 ** (sum(epoch > np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        if args.warmup_epochs > 0:
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch_float - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        else:
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch_float / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders(args):
    """Create log and model folder"""
    folders_util = [args.root_model, os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)




if __name__ == '__main__':
    main()
