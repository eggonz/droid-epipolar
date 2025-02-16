import cv2
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from geom.graph_utils import graph_to_edge_list
from geom.projective_ops import coords_grid, projective_transform
from geom import epipolar


class Logger:
    def __init__(self, name, scheduler, args, sum_freq=100):
        self.sum_freq = sum_freq
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.name = name
        self.scheduler = scheduler
        self.do_wandb = args.wandb
        self._init_wandb(args)

    def _print_training_status(self):
        if self.writer is None:
            self.writer = SummaryWriter('runs/%s' % self.name)
            print([k for k in self.running_loss])  # print metrics names

        lr = self.scheduler.get_lr().pop()
        metrics_data = {k: v/self.sum_freq for k, v in self.running_loss.items()}
        training_str = "[steps={:6d}, lr={:10.7f}] ".format(self.total_steps+1, lr)
        metrics_str = ", ".join([f"{k}={v:10.4f}" for k, v in metrics_data.items()])

        # print the training status
        print(training_str + metrics_str)

        for key in self.running_loss:
            val = self.running_loss[key] / self.sum_freq
            self.writer.add_scalar(key, val, self.total_steps)
            self.running_loss[key] = 0.0

    def _init_wandb(self, args):
        if not self.do_wandb:
            return
        to_resume = args.ckpt is not None
        if args.wandb_name is None:
            args.wandb_name = self.name
        if args.wandb_id is None:
            args.wandb_id = wandb.util.generate_id()
        wandb.login(key=args.wandb_key)
        wandb.init(
            name=args.wandb_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            id=args.wandb_id,
            resume="must" if to_resume else "allow",
        )
        wandb.config.update(args)

    def _log_wandb(self):
        if not self.do_wandb:
            return
        metrics_data = {k: v/self.sum_freq for k, v in self.running_loss.items()}
        metrics = {
            "lr": self.scheduler.get_lr().pop(),
            **metrics_data,
        }
        wandb.log(metrics, step=self.total_steps+1)

    def push(self, metrics):

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.sum_freq == self.sum_freq-1:
            self._log_wandb()
            self._print_training_status()
            self.running_loss = {}

        self.total_steps += 1

    def log_wandb_images(self, *args, **kwargs):
        """
        Args:
            images_dict: {name: [tensor_images]}
        """
        # src: https://docs.wandb.ai/guides/track/log/media
        if not self.do_wandb:
            return
        if self.total_steps % self.sum_freq == self.sum_freq-1:
            images_dict = build_example_images(*args, **kwargs)
            wandb_dict = {}
            for name, images in images_dict.items():
                if isinstance(images, list):
                    wandb_dict[name] = [wandb.Image(img) for img in images]
                else:
                    wandb_dict[name] = wandb.Image(images)
            wandb.log(wandb_dict, step=self.total_steps+1)

    def write_dict(self, results):
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def cart2polar(arr):
    """tensor: (H,W,2)"""

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()

    vx, vy = arr[..., 0], arr[..., 1]

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros((*vx.shape, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(vx, vy)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def normalize(arr, vmin=None, vmax=None):
    vmax = arr.max() if vmax is None else vmax
    vmin = arr.min() if vmin is None else vmin
    if vmax == vmin:
        arr = arr * 0
    else:
        arr = (arr - vmin) / (vmax - vmin)
    arr = arr.clip(0, 1).astype(np.float32)
    return arr


def build_example_images(images, poses_gt, depths_gt, intrinsics, poses_pred, depths_pred, residuals, graph, info):
    """
    Args:
        images: (B, N, 3, H, W)
        poses_gt: (B, N) se3
        depths_gt: (B, N, H, W)
        intrinsics: (B, N, 4)
        poses_pred: (B, N) se3
        depths_pred: (B, N, H, W)
        residuals: (B, E, h, w, 2)
        graph
        info:epipolar: (B, E, h, w, 1)
        info:ba_weights: (B, E, h, w, 2)
    """
    has_epipolar = 'epipolar' in info

    ii, jj, _ = graph_to_edge_list(graph)
    B, E = torch.randint(0, images.shape[0], (1,)).item(), torch.randint(0, ii.shape[0], (1,)).item()
    Ni, Nj = ii[E], jj[E]
    H, W = images.shape[-2:]

    image_src = images[B, Ni]  # (3, H, W)
    image_src = image_src.permute(1, 2, 0).byte()  # (H, W, 3)
    image_tgt = images[B, Nj]  # (3, H, W)
    image_tgt = image_tgt.permute(1, 2, 0).byte()  # (H, W, 3)
    depth_gt_src = depths_gt[B, Ni]  # (H, W)
    depth_gt_tgt = depths_gt[B, Nj]  # (H, W)
    depth_pred_src = depths_pred[B, Ni]  # (H, W)
    depth_pred_tgt = depths_pred[B, Nj]  # (H, W)

    residual_lowres = residuals[B, E]  # (h, w, 2)
    if has_epipolar:
        epipolar_lowres = info["epipolar"][B, E]  # (h, w, 1)
        exp_epipolar_lowres = torch.exp(epipolar_lowres)  # (h, w, 1)

    # flow
    grid = coords_grid(H, W, device=images.device)  # (H, W, 2)
    coords0, val0 = projective_transform(poses_gt, depths_gt, intrinsics, ii, jj)  # (B, E, H, W, 2), (B, E, H, W, 1)
    val0 = val0 * (depths_gt[:,ii] > 0).float().unsqueeze(dim=-1)
    coords1, val1 = projective_transform(poses_pred, depths_pred, intrinsics, ii, jj)  # (B, E, H, W, 2), (B, E, H, W, 1)
    v = (val0 * val1).squeeze(dim=-1)  # (B, E, H, W)
    c0, c1, v = coords0[B, E], coords1[B, E], v[B, E]  # (H, W, 2), (H, W, 2), (H, W)
    flow_gt = c0 - grid  # (H, W, 2)
    flow_pred = c1 - grid  # (H, W, 2)
    flow_diff = c1 - c0  # (H, W, 2)

    # epipolar full
    F = epipolar.get_fundamental_matrix_from_poses(poses_gt, graph, intrinsics)  # (B, E, 3, 3)
    ec = epipolar.epipolar_constraint(grid, coords1, F)  # (B, E, H, W)
    daec = epipolar.depth_aware_epipolar_constraint(ec, depths_gt[:,jj])  # (B, E, H, W)
    ec = ec[B, E]  # (H, W)
    daec = daec[B, E]  # (H, W)

    # BA weights, weighted norm
    ba_weights = info['ba_weights'][B, E]  # (h, w, 2)
    res_norm_weighted = (ba_weights * residual_lowres * residual_lowres).sum(dim=-1).sqrt()  # (h, w)

    # to numpy (H, W, C)
    image_src = image_src.detach().cpu().numpy()
    image_tgt = image_tgt.detach().cpu().numpy()
    depth_gt_src = depth_gt_src.detach().cpu().numpy()
    depth_gt_tgt = depth_gt_tgt.detach().cpu().numpy()
    depth_pred_src = depth_pred_src.detach().cpu().numpy()
    depth_pred_tgt = depth_pred_tgt.detach().cpu().numpy()
    residual_lowres = residual_lowres.detach().cpu().numpy()
    if has_epipolar:
        epipolar_lowres = epipolar_lowres.detach().cpu().numpy()
        exp_epipolar_lowres = exp_epipolar_lowres.detach().cpu().numpy()
    flow_gt = flow_gt.detach().cpu().numpy()
    flow_pred = flow_pred.detach().cpu().numpy()
    flow_diff = flow_diff.detach().cpu().numpy()
    ec = ec.detach().cpu().numpy()
    daec = daec.detach().cpu().numpy()
    ba_weights = ba_weights.detach().cpu().numpy()
    res_norm_weighted = res_norm_weighted.detach().cpu().numpy()

    # normalize
    vmax = max(depth_gt_src.max(), depth_gt_tgt.max(), depth_pred_src.max(), depth_pred_tgt.max())
    depth_gt_src = normalize(depth_gt_src, vmin=0, vmax=vmax)
    depth_gt_tgt = normalize(depth_gt_tgt, vmin=0, vmax=vmax)
    depth_pred_src = normalize(depth_pred_src, vmin=0, vmax=vmax)
    depth_pred_tgt = normalize(depth_pred_tgt, vmin=0, vmax=vmax)
    if has_epipolar:
        epipolar_lowres = normalize(epipolar_lowres)
        exp_epipolar_lowres = normalize(exp_epipolar_lowres)
    vmax = max(ec.max(), min(daec.max(), daec.mean() + 3 * daec.std()))
    ec_toscale = normalize(ec, vmin=0, vmax=vmax)
    daec_toscale = normalize(daec, vmin=0, vmax=vmax)
    ec = normalize(ec)
    daec = normalize(daec)
    res_norm_weighted = normalize(res_norm_weighted, vmin=0, vmax=2)

    # to polar
    residual_lowres = cart2polar(residual_lowres)
    flow_gt = cart2polar(flow_gt)
    flow_pred = cart2polar(flow_pred)
    flow_diff = cart2polar(flow_diff)
    ba_weights = cart2polar(ba_weights)

    # epipolar line example
    epiline_images = generate_epiline_images(image_src, image_tgt, grid, F[B, E], images.device)

    images_dict = {
        'images': [image_src, image_tgt],
        'depths_gt': [depth_gt_src, depth_gt_tgt],
        'depths_pred': [depth_pred_src, depth_pred_tgt],
        'residual_lowres': residual_lowres,
        'flow_gt': flow_gt,
        'flow_pred': flow_pred,
        'flow_diff': flow_diff,
        'epipolar_ec': ec,
        'epipolar_daec': daec,
        'epipolar_ec-toscale': ec_toscale,
        'epipolar_daec-toscale': daec_toscale,
        'epiline': epiline_images,
        'ba_weights': ba_weights,
        'residual_norm_weighted': res_norm_weighted,
    }
    if has_epipolar:
        images_dict['epipolar_lowres'] = epipolar_lowres
        images_dict['epipolar_lowres-exp'] = exp_epipolar_lowres
    return images_dict


def generate_epiline_images(image1, image2, grid, F, device):
    """ generates: [src_point, dst_line, epi_weight] """

    h, w = image1.shape[:2]  # 384, 512
    im_point = image1.copy()
    im_line = image2.copy()

    points = [
        (w // 4, h // 4, 1),
        (w // 4, h - h // 4, 1),
        (w - w // 4, h // 4, 1),
        (w - w // 4, h - h // 4, 1),
        (w // 2, h // 2, 1),
    ]
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
    ]

    for p, color in zip(points, colors):
        p = torch.tensor(p, device=device).float().unsqueeze(dim=-1)  # (3[xy1], 1)
        p = p / p[-1]  # homog
        grid3 = torch.cat([grid, torch.ones_like(grid[..., [0]])], dim=-1)  # (H, W, 3[xy1])
        pFp = grid3 @ F @ p  # (H, W, 1)

        epiline = F @ p  # (3[xy1],3[xy1]) @ (3[xy1],1) = (3[xy1],1)
        epiline = epiline.squeeze(dim=-1).detach().cpu().numpy()
        x1, x2 = 0, w
        a, b, c = epiline.flatten()  # epiline: ax + by + c = 0
        y1 = - (a * x1 + c) / b
        y2 = - (a * x2 + c) / b

        try:
            # clamp to canvas
            yy1 = max(0, min(h, y1))
            yy2 = max(0, min(h, y2))
            xx1 = w - w / (y2 - y1 + 1e-10) * (y2 - yy1)
            xx2 = w / (y2 - y1 + 1e-10) * (yy2 - y1)

            im_point = cv2.circle(im_point.copy(), (int(p[0]), int(p[1])), 5, color, -1)

            is_within_limits = 0 <= xx1 <= w and 0 <= xx2 <= w and 0 <= yy1 <= h and 0 <= yy2 <= h
            is_nan = np.isnan(xx1) or np.isnan(xx2) or np.isnan(yy1) or np.isnan(yy2)
            is_inf = np.isinf(xx1) or np.isinf(xx2) or np.isinf(yy1) or np.isinf(yy2)
            if is_within_limits and not is_nan and not is_inf:
                im_line = cv2.line(im_line.copy(), (int(xx1), int(yy1)), (int(xx2), int(yy2)), color, 2)
        except Exception as e:
            print(f"[ERR] Error in epiline generation (x1={x1}, x2={x2}, y1={y1}, y2={y2}).", e)

    # im_weights: epiweights of last point (center)
    im_weights = pFp.squeeze(dim=-1).detach().cpu().numpy()
    im_weights = np.abs(im_weights)  # |pFp|
    im_weights = normalize(im_weights)
    im_weights = cv2.applyColorMap((im_weights * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    if is_within_limits and not is_nan and not is_inf:
        im_weights = cv2.line(im_weights, (int(xx1), int(yy1)), (int(xx2), int(yy2)), color, 2)

    try:
        cv2_src, cv2_dst = cv2_epilines(image1, image2)
    except Exception as e:
        print('[ERR]', e)
        cv2_src, cv2_dst = image1.copy(), image2.copy()

    return [im_point, im_line, im_weights, cv2_src, cv2_dst]


# cv2.epipolar

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def cv2_epilines(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    try:
        F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
    except Exception as e:
        print('[ERR]', e)
        return img1, img2

    if mask is None:
        return img1, img2
    
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    return img5, img3
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.savefig("cv2_epilines.png")  # example for debug
    plt.close()
