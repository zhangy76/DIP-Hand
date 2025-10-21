import numpy as np
import torch
import torch.nn.functional as F


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def batch_euler2matzxy(angle):
    """
    Convert euler angles to rotation matrix.
    Args:
        angle: [N, 3], rotation angle along 3 axis (in radians)
    Returns:
        Rotation: [N, 3, 3], matrix corresponding to the euler angles
    """
    # obtain the batch size
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack(
        [cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1
    ).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack(
        [cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1
    ).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack(
        [ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1
    ).reshape(B, 3, 3)

    rotMat = ymat @ xmat @ zmat
    return rotMat


def coordtrans(rotmat, local2global, transtype):

    batchsize = rotmat.shape[0]
    local2global = local2global.clone()
    if transtype == 1:
        rotmat[:, 1:] = torch.einsum(
            "...ij,...jk->...ik", rotmat[:, 1:], local2global[:batchsize]
        )
        rotmat[:, 1:] = torch.einsum(
            "...ij,...jk->...ik",
            local2global[:batchsize].transpose(2, 3),
            rotmat[:, 1:],
        )
    else:
        rotmat[:, 1:] = torch.einsum(
            "...ij,...jk->...ik",
            rotmat[:, 1:],
            local2global[:batchsize].transpose(2, 3),
        )
        rotmat[:, 1:] = torch.einsum(
            "...ij,...jk->...ik", local2global[:batchsize], rotmat[:, 1:]
        )

    return rotmat


def rotmat2eulerzxy(rotmat):
    """Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (N,3,3)
    Returns
    -------
    angle : (N,3)
       Rotations in radians around z, x, y axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
      [cos(y)*cos(z) + sin(x)*sin(y)*sin(z), cos(z)*sin(x)*sin(y) - cos(y)*sin(z), cos(x)*sin(y)],
      [                       cos(x)*sin(z),                        cos(x)*cos(z),      - sin(x)],
      [cos(y)*sin(x)*sin(z) - cos(z)*sin(y), sin(y)*sin(z) + cos(y)*cos(z)*sin(x), cos(x)*cos(y)]
    """
    N = rotmat.size()[0]
    cx_thresh = torch.ones(N).type(torch.float32).to(rotmat.device) * 1e-8

    eulerangle = torch.zeros(N, 3).type(torch.float32).to(rotmat.device)

    r11, r12, r13 = rotmat[:, 0, 0], rotmat[:, 0, 1], rotmat[:, 0, 2]
    r21, r22, r23 = rotmat[:, 1, 0], rotmat[:, 1, 1], rotmat[:, 1, 2]
    _, _, r33 = rotmat[:, 2, 0], rotmat[:, 2, 1], rotmat[:, 2, 2]

    # cx
    cx = torch.sqrt(torch.pow(r21, 2) + torch.pow(r22, 2))
    # x = atan(sin(x),con(x))
    eulerangle[:, 0] = torch.atan2(-r23, cx)  # [-pi,pi]

    # c>cx_thresh
    eulerangle[cx > cx_thresh, 2] = torch.atan2(
        r21[cx > cx_thresh], r22[cx > cx_thresh]
    )
    eulerangle[cx > cx_thresh, 1] = torch.atan2(
        r13[cx > cx_thresh], r33[cx > cx_thresh]
    )

    # cy<=cy_thresh
    eulerangle[cx <= cx_thresh, 2] = torch.atan2(
        -r12[cx <= cx_thresh], r11[cx <= cx_thresh]
    )

    return eulerangle


def crop2expandsquare_zeros(img, bbox, expand_ratio):

    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]
    if C == 0:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif C == 1:
        img = np.repeat(img, 3, axis=2)
    C = 3

    scale = 2 * (np.random.rand(4) - 0.5) * expand_ratio + 1
    # update bbox using length to center and scale
    x1_new = max(int(bbox[0] - bbox[2] * scale[0]), 0)
    y1_new = max(int(bbox[1] - bbox[3] * scale[1]), 0)
    x2_new = min(int(bbox[0] + bbox[2] * scale[2]), W)
    y2_new = min(int(bbox[1] + bbox[3] * scale[3]), H)

    Len_max = np.amax([y2_new - y1_new, x2_new - x1_new])

    x1_new_square = int((Len_max - x2_new + x1_new) / 2)
    y1_new_square = int((Len_max - y2_new + y1_new) / 2)
    x2_new_square = int((Len_max + x2_new - x1_new) / 2)
    y2_new_square = int((Len_max + y2_new - y1_new) / 2)

    # square image
    img_cropped_square = np.zeros([Len_max, Len_max, C], dtype=np.uint8)

    img_cropped_square[y1_new_square:y2_new_square, x1_new_square:x2_new_square, :] = (
        img[y1_new:y2_new, x1_new:x2_new]
    )

    offset_Pts = np.array([[x1_new - x1_new_square, y1_new - y1_new_square]])
    offset = [x1_new_square, y1_new_square]
    return img_cropped_square, offset_Pts, offset
