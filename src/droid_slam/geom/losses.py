from collections import OrderedDict

import numpy as np
from lietorch import SO3, SE3, Sim3

from .epipolar import get_fundamental_matrix_from_poses, epipolar_constraint, depth_aware_epipolar_constraint
from .graph_utils import graph_to_edge_list
from .projective_ops import coords_grid, projective_transform


def pose_metrics(dE):
    """ Translation/Rotation/Scaling metrics from Sim3 """
    t, q, s = dE.data.split([3, 4, 1], -1)
    ang = SO3(q).log().norm(dim=-1)

    # convert radians to degrees
    r_err = (180 / np.pi) * ang
    t_err = t.norm(dim=-1)
    s_err = (s - 1.0).abs()
    return r_err, t_err, s_err


def fit_scale(Ps, Gs):
    b = Ps.shape[0]
    t1 = Ps.data[...,:3].detach().reshape(b, -1)
    t2 = Gs.data[...,:3].detach().reshape(b, -1)

    s = (t1*t2).sum(-1) / ((t2*t2).sum(-1) + 1e-8)
    return s


def geodesic_loss(Ps, Gs, graph, gamma=0.9, do_scale=True):
    """ Loss function for training network """

    # relative pose
    ii, jj, kk = graph_to_edge_list(graph)
    dP = Ps[:,jj] * Ps[:,ii].inv()

    n = len(Gs)
    geodesic_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        dG = Gs[i][:,jj] * Gs[i][:,ii].inv()

        if do_scale:
            s = fit_scale(dP, dG)
            dG = dG.scale(s[:,None])
        
        # pose error
        d = (dG * dP.inv()).log()

        if isinstance(dG, SE3):
            tau, phi = d.split([3,3], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() + 
                phi.norm(dim=-1).mean())

        elif isinstance(dG, Sim3):
            tau, phi, sig = d.split([3,3,1], dim=-1)
            geodesic_loss += w * (
                tau.norm(dim=-1).mean() + 
                phi.norm(dim=-1).mean() + 
                0.05 * sig.norm(dim=-1).mean())
            
        dE = Sim3(dG * dP.inv()).detach()
        r_err, t_err, s_err = pose_metrics(dE)

    metrics = {
        'rot_error': r_err.mean().item(),
        'tr_error': t_err.mean().item(),
        'bad_rot': (r_err < .1).float().mean().item(),
        'bad_tr': (t_err < .01).float().mean().item(),
        'geod_loss': geodesic_loss.item(),
    }
    return geodesic_loss, metrics


def residual_loss(residuals, gamma=0.9, ec_weights=None):
    """ loss on system residuals 
    
    Args:
        residuals: list of residual tensors [B,E,H,W,2]
        gamma: discount factor
        ec_weights: list of epipolar constraint weights [B,E,H,W,1]
    """
    residual_loss = 0.0
    residual_loss_ec = 0.0
    n = len(residuals)

    for i in range(n):
        w = gamma ** (n - i - 1)
        residual_loss += w * residuals[i].abs().mean()
        if ec_weights is not None:
            residual_loss_ec += w * (ec_weights[i] * residuals[i]).abs().mean()

    res_loss = residual_loss_ec if ec_weights is not None else residual_loss
    metrics = {
        'res_loss': res_loss.item(),
        'residual': residual_loss.item(),
    }
    if ec_weights is not None:
        metrics['residual_ec'] = residual_loss_ec.item()
    return res_loss, metrics


def flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph, gamma=0.9):
    """ optical flow loss """

    N = Ps.shape[1]
    graph = OrderedDict()
    for i in range(N):
        graph[i] = [j for j in range(N) if abs(i-j)==1]

    ii, jj, kk = graph_to_edge_list(graph)
    coords0, val0 = projective_transform(Ps, disps, intrinsics, ii, jj)
    val0 = val0 * (disps[:,ii] > 0).float().unsqueeze(dim=-1)

    n = len(poses_est)
    flow_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        coords1, val1 = projective_transform(poses_est[i], disps_est[i], intrinsics, ii, jj)

        v = (val0 * val1).squeeze(dim=-1)
        epe = v * (coords1 - coords0).norm(dim=-1)
        flow_loss += w * epe.mean()

    epe = epe.reshape(-1)[v.reshape(-1) > 0.5]
    metrics = {
        'f_error': epe.mean().item(),
        '1px': (epe<1.0).float().mean().item(),
        'flow_loss': flow_loss.item(),
    }
    return flow_loss, metrics


def epipolar_loss(gt_pose, gt_depth, pred_poses, pred_depths, intrinsics, graph, gamma=0.9, depth_aware=False):
          
    ii, jj, _ = graph_to_edge_list(graph)
    H, W = gt_depth.shape[-2:]  # [B, N, H, W]
    coords0 = coords_grid(H, W, device=gt_depth.device)  # [H, W, 2]

    F = get_fundamental_matrix_from_poses(gt_pose, graph, intrinsics)  # [B, E, 3, 3]

    n = len(pred_poses)
    ec_loss = 0.0
    daec_loss = 0.0

    for i in range(n):
        w = gamma ** (n - i - 1)
        coords1, valid = projective_transform(pred_poses[i], pred_depths[i], intrinsics, ii, jj)  # [B, E, H, W, 2], [B, E, H, W, 1]
        valid = valid.squeeze(dim=-1).float()  # [B, E, H, W]

        ec = epipolar_constraint(coords0, coords1, F)  # [B, E, H, W]
        ec_loss += w * (valid * ec).mean()

        daec = depth_aware_epipolar_constraint(ec, gt_depth[:,jj])  # [B, E, H, W]
        daec_loss += w * (valid * daec).mean()

    epi_loss = daec_loss if depth_aware else ec_loss
    metrics = {
        'ec': ec_loss.item(),
        'daec': daec_loss.item(),
        'epi_loss': epi_loss.item(),
    }
    return epi_loss, metrics
