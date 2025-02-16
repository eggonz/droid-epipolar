import kornia
import torch

from .graph_utils import graph_to_edge_list
from .projective_ops import extract_intrinsics


def _build_intrinsic_matrix(intrinsics):
    """
    Args:
        intrinsics: tensor [B, N, 4]
    Return:
        K: tensor [B, N, 3, 3]
    """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    o, i = torch.zeros_like(fx), torch.ones_like(fx)
    K = torch.stack([fx, o, cx, o, fy, cy, o, o, i], dim=-1)
    K = K.view(*K.shape[:2], 3, 3).to(intrinsics.device)
    return K


def get_fundamental_matrix_from_poses(Ps, graph, intrinsics):
    """ Computes fundamental matrix from SE3 poses.

    Args:
        Ps: tensor [B, N] se3
        graph: OrderedDict {src: [dst,]}
        intrinsics: tensor [B, N, 4]
    Return:
        F: tensor [B, E, 3, 3]
    """
    device = intrinsics.device
    ii, jj, _ = graph_to_edge_list(graph)  # [E,]
    Ps = Ps.to(device)
    Ps1 = Ps[:, ii]  # [B, E] se3
    Ps2 = Ps[:, jj]
    R1, t1 = Ps1.matrix()[..., :3, :3], Ps1.matrix()[..., :3, [3]]
    R2, t2 = Ps2.matrix()[..., :3, :3], Ps2.matrix()[..., :3, [3]]
    E = kornia.geometry.epipolar.essential_from_Rt(R1, t1, R2, t2)  # [B, E, 3, 3]
    E = E.to(device)
    K1 = _build_intrinsic_matrix(intrinsics[:, ii])  # [B, E, 3, 3]
    K2 = _build_intrinsic_matrix(intrinsics[:, jj])
    F = kornia.geometry.epipolar.fundamental_from_essential(E, K1, K2)  # [B, E, 3, 3]
    return F.to(device)


# def get_fundamental_matrix_from_matches(image1, image2):
#     """ Computes Fundamental matrix from mathes between two images, using opencv.

#     Args:
#         image1: tensor [B, C, H, W]
#         image2: tensor [B, C, H, W]
#         intrinsics: tensor [B, 4]

#     Return:
#         F: tensor [B, 3, 3] at image1.device
#     """
#     device = image1.device
#     image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
#     image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
#     # find the keypoints and descriptors with SIFT
#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(image1,None)
#     kp2, des2 = sift.detectAndCompute(image2,None)
#     # FLANN parameters
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=10)
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params,search_params)
#     matches = flann.knnMatch(des1,des2,k=2)
#     good_matches = [m for m,n in matches if m.distance < 0.7*n.distance]
#     if len(good_matches) < 10:
#         print("<10 matches found, using BF matcher instead")
#         bf = cv2.BFMatcher()
#         matches = bf.knnMatch(des1,des2, k=2)
#         good_matches = [m for m,n in matches if m.distance < 0.65*n.distance]
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
#     F, mask = cv2.findFundamentalMat(src_pts,dst_pts,cv2.FM_LMEDS)
#     if mask is None:
#         print("No matches found")
#         return None
#     return F.to(device)


def epipolar_constraint(c0, c1, f):
    """
    |p1.T @ F @ p0|, ideally ec=|pFp|=0
    Expects fixed grid coordinates at source and predicted coordinates at destination, and fundamental matrix from source to destination computed from GT poses.

    Args:
        c0: tensor [h, w, 2(x,y)] or [..., h, w, 2(x,y)], grid coords at source
        c1: tensor [..., h, w, 2(x,y)], predicted coords at destination or target, e.g. [B, E, h, w, 2] or [B, h, w, 2]
        f: tensor [..., 3, 3], e.g. [B, E, 3, 3] or [B, 3, 3]
    Return:
        ec: tensor [..., h, w]
    """
    coords0 = torch.cat((c0, torch.ones_like(c0[..., :1])), dim=-1)  # [..., 3(x,y,1)]
    coords1 = torch.cat((c1, torch.ones_like(c1[..., :1])), dim=-1)  # [..., 3(x,y,1)]

    orig_shape = coords1.shape  #  e.g. [B, E, h, w, 3] or [B, h, w, 3]
    h, w = coords1.shape[-3:-1]
    coords1 = coords1.view(-1, h, w, 3)  # [b, h, w, 3]
    f = f.view(-1, 3, 3)  # [b, 3, 3]

    einsum_str = 'bij,hwj->bhwi' if len(coords0.shape) == 3 else 'bij,bhwj->bhwi'
    Fp = torch.einsum(einsum_str, f, coords0)  # [b, 3, 3] x [(b, )h, w, 3] -> [b, h, w, 3] epiline
    Fp = Fp / Fp.norm(p=2, dim=-1, keepdim=True)  # normalize epiline
    coords1 = coords1 / coords1.norm(p=2, dim=-1, keepdim=True)  # normalize coords
    pFp = (coords1 * Fp).sum(dim=-1)  # [b, h, w, 3] x [b, h, w, 3] -> [b, h, w]
    pFp = pFp.abs()  # [b, h, w]
    pFp = pFp.view(*orig_shape[:-1])  # e.g. [B, E, h, w] or [B, h, w]
    return pFp


def depth_aware_epipolar_constraint(ec, d, max_depth=100):
    """
    Compute depth-aware epipolar constraint.
    Expects output from `epipolar_constraint` and depths at target frame.

    Args:
        ec: tensor [..., h, w], outputof `epipolar_constraint`
        d: tensor [..., h, w] (inverse) depths, e.g. [B, E, h, w] or [B, h, w]
    Return:
        daec: tensor [..., h, w]
    """
    # d is inverse depth, set limit to avoid exploding gradients
    depth = 1 / d
    depth = max_depth * torch.tanh(depth / max_depth)
    daec = ec * depth
    return daec


def exponential_ec(ec, sigma=1):
    """
    exp(ec/sigma)=exp(|pFp|/sigma), where ec=|pFp| is the epipolar constraint

    Args:
        ec: tensor or list [..., h, w, 1], e.g. [B, E, h, w, 1] or [B, h, w, 1]
    Return:
        exp_ec: tensor [..., h, w, 1]
    """
    if isinstance(ec, list):
        ec = torch.stack(ec, dim=0)
    exp_ec = torch.exp(ec / sigma)
    return exp_ec
