import sys
from datetime import datetime
from functools import partial

sys.path.append('droid_slam')

import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
from geom import losses, epipolar
from geom.graph_utils import build_frame_graph, graph_to_edge_list

# network
from droid_net import DroidNet
from logger import Logger

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm


def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def train(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    setup_ddp(gpu, args)
    rng = np.random.default_rng(12345)

    N = args.n_frames
    model = DroidNet()
    model.cuda()
    model.train()

    model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    # fetch dataloader
    db = dataset_factory(['tartan'], datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=gpu)

    train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=2)

    # fetch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False)

    logger = Logger(args.name, scheduler, args, sum_freq=args.log_freq)
    should_keep_training = True
    total_steps = 0

    ec_fn = epipolar.epipolar_constraint
    daec_fn = lambda c0, c1, f, d: epipolar.depth_aware_epipolar_constraint(ec_fn(c0, c1, f), d)

    time_start = datetime.now()

    while should_keep_training:
        for i_batch, item in enumerate(tqdm(train_loader, desc='Training')):
            optimizer.zero_grad()

            images, poses, disps, intrinsics = [x.to('cuda') for x in item]
            # `instrinsics` are all the same ((intrinsics == intrinsics[0]).all()) [see droid_slam.data_reader.tartan.TartanAir.calib_read()]

            # convert poses w2c -> c2w
            Ps = SE3(poses).inv()
            Gs = SE3.IdentityLike(Ps)

            # randomize frame graph
            if np.random.rand() < 0.5:
                # build_frame_graph builds graph using intrinsics/8 (1/8 resolution)
                graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)
            
            else:
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]
            
            # fix first to camera poses
            Gs.data[:,0] = Ps.data[:,0].clone()
            Gs.data[:,1:] = Ps.data[:,[1]].clone()
            disp0 = torch.ones_like(disps[:,:,3::8,3::8])  # disps full-res; disp0 1/8 resolution
            intrinsics0 = intrinsics / 8.0  # intrinsics full-res, intrinsics0 1/8 resolution

            F = epipolar.get_fundamental_matrix_from_poses(Ps, graph, intrinsics0)  # gt fundamental matrix, 1/8 resolution
            if not args.res_loss_ec and not args.ba_update_ec:
                epipolar_fn = None
            elif args.use_daec:
                # disp0 is updated in the loop
                pass
            else:
                epipolar_fn = partial(ec_fn, f=F)  # same across loop

            # perform random restarts
            r = 0
            while r < args.restart_prob:
                r = rng.random()

                if (args.res_loss_ec or args.ba_update_ec) and args.use_daec:
                    _, jj, _ = graph_to_edge_list(graph)
                    epipolar_fn = partial(daec_fn, f=F, d=disp0[:,jj])

                poses_est, disps_est, residuals, ec_list, info = model(
                    Gs, images, disp0, intrinsics0, graph, num_steps=args.iters, fixedp=2,
                    ba_update_ec=args.ba_update_ec, res_loss_ec=args.res_loss_ec, epipolar_fn=epipolar_fn)
                # disps_est full-res (upsampled); residuals, ec_list 1/8 resolution

                geod_loss, geod_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False)
                if args.res_loss_ec:
                    exp_ec = epipolar.exponential_ec(ec_list, sigma=args.ec_sigma)
                    res_loss, res_metrics = losses.residual_loss(residuals, ec_weights=exp_ec)
                else:
                    res_loss, res_metrics = losses.residual_loss(residuals)
                flo_loss, flo_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph)
                loss = args.w1 * geod_loss + args.w2 * res_loss + args.w3 * flo_loss

                if args.ec_loss:
                    epi_loss, epi_metrics = losses.epipolar_loss(Ps, disps, poses_est, disps_est, intrinsics, graph, depth_aware=args.use_daec)
                    loss = loss + args.w4 * epi_loss
                else:
                    # still compute and log ec and daec metrics
                    _, epi_metrics = losses.epipolar_loss(Ps, disps, poses_est, disps_est, intrinsics, graph)
                    epi_metrics.pop('epi_loss')

                loss.backward()

                Gs = poses_est[-1].detach()
                disp0 = disps_est[-1][:,:,3::8,3::8].detach()

            metrics = {}
            metrics.update(geod_metrics)
            metrics.update(res_metrics)
            metrics.update(flo_metrics)
            metrics.update(epi_metrics)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            
            total_steps += 1

            if gpu == 0:
                logger.log_wandb_images(images, Ps, disps, intrinsics, poses_est[-1], disps_est[-1], residuals[-1], graph, info)
                logger.push(metrics)

            if total_steps % 10000 == 0 and gpu == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

            if total_steps >= args.steps:
                should_keep_training = False
                break

    time_end = datetime.now()
    if gpu == 0:
        print(f'Training time: {time_end - time_start}')

    dist.destroy_process_group()
                

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='datasets/TartanAir', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=4)

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)

    parser.add_argument('--w1', type=float, default=10.0, help='geodesic loss lambda weight')  # 2.0 x w=10 = 20
    parser.add_argument('--w2', type=float, default=0.01, help='residual loss lambda weight')  # 2.0 x w=0.01 = 0.02
    parser.add_argument('--w3', type=float, default=0.05, help='flow loss lambda weight')  # 270.0 x w=0.05 = 13.5
    parser.add_argument('--w4', type=float, default=10, help='epipolar loss lambda weight')  # 0.003 x w=10 = 0.03
    parser.add_argument('--ec-sigma', type=float, default=1.0, help='exp(|pFp|/sigma) in epipolar constraint')

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    # wandb
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='droidslam_epipolar')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_mode', type=str, default='online')
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, default=None, help='name your wandb run, defaults to --name')

    parser.add_argument('--log_freq', type=int, default=100)

    # experiment args
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--use-daec', action='store_true', help='use depth-aware epipolar constraint')
    parser.add_argument('--ec-loss', action='store_true', help='add epipolar constraint loss')
    parser.add_argument('--res-loss-ec', action='store_true', help='whether to add exponential epipolar constraint as weights to residual loss')
    parser.add_argument('--ba-update-ec', action='store_true', help='whether to include epipolar constraint as input to the BA update block')

    args = parser.parse_args()

    args.world_size = args.gpus
    print(args)

    import os
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    args = parser.parse_args()
    args.world_size = args.gpus

    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

