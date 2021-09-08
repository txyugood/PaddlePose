import math

import cv2
import numpy as np

def fliplr_joints(joints_3d, joints_3d_visible, img_width, flip_pairs):
    """Flip human joints horizontally.

    Note:
        num_keypoints: K

    Args:
        joints_3d (np.ndarray([K, 3])): Coordinates of keypoints.
        joints_3d_visible (np.ndarray([K, 1])): Visibility of keypoints.
        img_width (int): Image width.
        flip_pairs (list[tuple()]): Pairs of keypoints which are mirrored
            (for example, left ear -- right ear).

    Returns:
        tuple: Flipped human joints.

        - joints_3d_flipped (np.ndarray([K, 3])): Flipped joints.
        - joints_3d_visible_flipped (np.ndarray([K, 1])): Joint visibility.
    """

    assert len(joints_3d) == len(joints_3d_visible)
    assert img_width > 0

    joints_3d_flipped = joints_3d.copy()
    joints_3d_visible_flipped = joints_3d_visible.copy()

    # Swap left-right parts
    for left, right in flip_pairs:
        joints_3d_flipped[left, :] = joints_3d[right, :]
        joints_3d_flipped[right, :] = joints_3d[left, :]

        joints_3d_visible_flipped[left, :] = joints_3d_visible[right, :]
        joints_3d_visible_flipped[right, :] = joints_3d_visible[left, :]

    # Flip horizontally
    joints_3d_flipped[:, 0] = img_width - 1 - joints_3d_flipped[:, 0]
    joints_3d_flipped = joints_3d_flipped * joints_3d_visible_flipped

    return joints_3d_flipped, joints_3d_visible_flipped



class LoadImageFromFile:
    """Loading image from file.

    Args:
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Loading image from file."""
        image_file = results['image_file']
        img = cv2.imread(image_file)
        if self.channel_order:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            raise ValueError(f'Fail to read {image_file}')

        results['img'] = img
        return results

class TopDownRandomFlip:
    """Data augmentation with random image flip.

    Required keys: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'ann_info'.
    Modifies key: 'img', 'joints_3d', 'joints_3d_visible', 'center' and
    'flipped'.

    Args:
        flip (bool): Option to perform random flip.
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        center = results['center']

        # A flag indicating whether the image is flipped,
        # which can be used by child class.
        flipped = False
        if np.random.rand() <= self.flip_prob:
            flipped = True
            img = img[:, ::-1, :]
            joints_3d, joints_3d_visible = fliplr_joints(
                joints_3d, joints_3d_visible, img.shape[1],
                results['ann_info']['flip_pairs'])
            center[0] = img.shape[1] - center[0] - 1

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['center'] = center
        results['flipped'] = flipped

        return results

class TopDownGetRandomScaleRotation:
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'. Modifies key: 'scale' and 'rotation'.

    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=40, scale_factor=0.5, rot_prob=0.6):
        self.rot_factor = rot_factor
        self.scale_factor = scale_factor
        self.rot_prob = rot_prob

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['scale']

        sf = self.scale_factor
        rf = self.rot_factor

        s_factor = np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        s = s * s_factor

        r_factor = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        r = r_factor if np.random.rand() <= self.rot_prob else 0

        results['scale'] = s
        results['rotation'] = r

        return results

def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        matrix (np.ndarray): A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return matrix


def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.

    Returns:
        matrix (np.ndarray[..., 2]): Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
        mat.T).reshape(shape)

def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt

def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.

    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform

    Returns:
        np.ndarray: Transformed points.
    """
    assert len(pt) == 2
    new_pt = np.array(trans_mat) @ np.array([pt[0], pt[1], 1.])

    return new_pt



class TopDownAffine:
    """Affine transform the image to make input.

    Required keys:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'. Modified keys:'img', 'joints_3d', and
    'joints_3d_visible'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    def __call__(self, results):
        image_size = results['ann_info']['image_size']

        img = results['img']
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']
        c = results['center']
        s = results['scale']
        r = results['rotation']

        if self.use_udp:
            trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
            img = cv2.warpAffine(
                img,
                trans, (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR)
            joints_3d[:, 0:2] = \
                warp_affine_joints(joints_3d[:, 0:2].copy(), trans)
        else:
            trans = get_affine_transform(c, s, r, image_size)
            img = cv2.warpAffine(
                img,
                trans, (int(image_size[0]), int(image_size[1])),
                flags=cv2.INTER_LINEAR)
            for i in range(results['ann_info']['num_joints']):
                if joints_3d_visible[i, 0] > 0.0:
                    joints_3d[i,
                              0:2] = affine_transform(joints_3d[i, 0:2], trans)

        results['img'] = img
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible

        return results

class NormalizeTensor:
    """Normalize the Tensor image (CxHxW), with mean and std.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        mean (list[float]): Mean values of 3 channels.
        std (list[float]): Std values of 3 channels.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        img = results['img']
        img = img.astype(np.float32, copy=False) / 255.0
        img -= self.mean
        img /= self.std
        img = np.transpose(img, [2, 0, 1])
        results['img'] = img

        return results

class TopDownGenerateTarget:
    """Generate the target heatmap.

    Required keys: 'joints_3d', 'joints_3d_visible', 'ann_info'.
    Modified keys: 'target', and 'target_weight'.

    Args:
        sigma: Sigma of heatmap gaussian for 'MSRA' approach.
        kernel: Kernel of heatmap gaussian for 'Megvii' approach.
        encoding (str): Approach to generate target heatmaps.
            Currently supported approaches: 'MSRA', 'Megvii', 'UDP'.
            Default:'MSRA'

        unbiased_encoding (bool): Option to use unbiased
            encoding methods.
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        keypoint_pose_distance: Keypoint pose distance for UDP.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
        target_type (str): supported targets: 'GaussianHeatmap',
            'CombinedTarget'. Default:'GaussianHeatmap'
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self,
                 sigma=2,
                 kernel=(11, 11),
                 valid_radius_factor=0.0546875,
                 target_type='GaussianHeatmap',
                 encoding='MSRA',
                 unbiased_encoding=False):
        self.sigma = sigma
        self.unbiased_encoding = unbiased_encoding
        self.kernel = kernel
        self.valid_radius_factor = valid_radius_factor
        self.target_type = target_type
        self.encoding = encoding

    def _msra_generate_target(self, cfg, joints_3d, joints_3d_visible, sigma):
        """Generate the target heatmap via "MSRA" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            sigma: Sigma of heatmap gaussian
        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        target = np.zeros((num_joints, H, W), dtype=np.float32)

        # 3-sigma rule
        tmp_size = sigma * 3

        if self.unbiased_encoding:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                # Check that any part of the gaussian is in-bounds
                ul = [mu_x - tmp_size, mu_y - tmp_size]
                br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] == 0:
                    continue

                x = np.arange(0, W, 1, np.float32)
                y = np.arange(0, H, 1, np.float32)
                y = y[:, None]

                if target_weight[joint_id] > 0.5:
                    target[joint_id] = np.exp(-((x - mu_x)**2 +
                                                (y - mu_y)**2) /
                                              (2 * sigma**2))
        else:
            for joint_id in range(num_joints):
                target_weight[joint_id] = joints_3d_visible[joint_id, 0]

                feat_stride = image_size / [W, H]
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
                    target_weight[joint_id] = 0

                if target_weight[joint_id] > 0.5:
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, None]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized,
                    # we want the center value to equal 1
                    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], W)
                    img_y = max(0, ul[1]), min(br[1], H)

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def _megvii_generate_target(self, cfg, joints_3d, joints_3d_visible,
                                kernel):
        """Generate the target heatmap via "Megvii" approach.

        Args:
            cfg (dict): data config
            joints_3d: np.ndarray ([num_joints, 3])
            joints_3d_visible: np.ndarray ([num_joints, 3])
            kernel: Kernel of heatmap gaussian
        Returns:
            tuple: A tuple containing targets.

            - target: Target heatmaps.
            - target_weight: (1: visible, 0: invisible)
        """

        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        W, H = cfg['heatmap_size']
        heatmaps = np.zeros((num_joints, H, W), dtype='float32')
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)

        for i in range(num_joints):
            target_weight[i] = joints_3d_visible[i, 0]

            if target_weight[i] < 1:
                continue

            target_y = int(joints_3d[i, 1] * H / image_size[1])
            target_x = int(joints_3d[i, 0] * W / image_size[0])

            if (target_x >= W or target_x < 0) \
                    or (target_y >= H or target_y < 0):
                target_weight[i] = 0
                continue

            heatmaps[i, target_y, target_x] = 1
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
            maxi = heatmaps[i, target_y, target_x]

            heatmaps[i] /= maxi / 255

        return heatmaps, target_weight

    def _udp_generate_target(self, cfg, joints_3d, joints_3d_visible, factor,
                             target_type):
        """Generate the target heatmap via 'UDP' approach. Paper ref: Huang et
        al. The Devil is in the Details: Delving into Unbiased Data Processing
        for Human Pose Estimation (CVPR 2020).

        Note:
            num keypoints: K
            heatmap height: H
            heatmap width: W
            num target channels: C
            C = K if target_type=='GaussianHeatmap'
            C = 3*K if target_type=='CombinedTarget'

        Args:
            cfg (dict): data config
            joints_3d (np.ndarray[K, 3]): Annotated keypoints.
            joints_3d_visible (np.ndarray[K, 3]): Visibility of keypoints.
            factor (float): kernel factor for GaussianHeatmap target or
                valid radius factor for CombinedTarget.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Heatmap target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).

        Returns:
            tuple: A tuple containing targets.

            - target (np.ndarray[C, H, W]): Target heatmaps.
            - target_weight (np.ndarray[K, 1]): (1: visible, 0: invisible)
        """
        num_joints = cfg['num_joints']
        image_size = cfg['image_size']
        heatmap_size = cfg['heatmap_size']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']

        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d_visible[:, 0]

        if target_type.lower() == 'GaussianHeatmap'.lower():
            target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = factor * 3

            # prepare for gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]

            for joint_id in range(num_joints):
                feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
                mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                mu_x_ac = joints_3d[joint_id][0] / feat_stride[0]
                mu_y_ac = joints_3d[joint_id][1] / feat_stride[1]
                x0 = y0 = size // 2
                x0 += mu_x_ac - mu_x
                y0 += mu_y_ac - mu_y
                g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        elif target_type.lower() == 'CombinedTarget'.lower():
            target = np.zeros(
                (num_joints, 3, heatmap_size[1] * heatmap_size[0]),
                dtype=np.float32)
            feat_width = heatmap_size[0]
            feat_height = heatmap_size[1]
            feat_x_int = np.arange(0, feat_width)
            feat_y_int = np.arange(0, feat_height)
            feat_x_int, feat_y_int = np.meshgrid(feat_x_int, feat_y_int)
            feat_x_int = feat_x_int.flatten()
            feat_y_int = feat_y_int.flatten()
            # Calculate the radius of the positive area in classification
            #   heatmap.
            valid_radius = factor * heatmap_size[1]
            feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
            for joint_id in range(num_joints):
                mu_x = joints_3d[joint_id][0] / feat_stride[0]
                mu_y = joints_3d[joint_id][1] / feat_stride[1]
                x_offset = (mu_x - feat_x_int) / valid_radius
                y_offset = (mu_y - feat_y_int) / valid_radius
                dis = x_offset**2 + y_offset**2
                keep_pos = np.where(dis <= 1)[0]
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id, 0, keep_pos] = 1
                    target[joint_id, 1, keep_pos] = x_offset[keep_pos]
                    target[joint_id, 2, keep_pos] = y_offset[keep_pos]
            target = target.reshape(num_joints * 3, heatmap_size[1],
                                    heatmap_size[0])
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")

        if use_different_joint_weights:
            target_weight = np.multiply(target_weight, joint_weights)

        return target, target_weight

    def __call__(self, results):
        """Generate the target heatmap."""
        joints_3d = results['joints_3d']
        joints_3d_visible = results['joints_3d_visible']

        assert self.encoding in ['MSRA', 'Megvii', 'UDP']

        if self.encoding == 'MSRA':
            if isinstance(self.sigma, list):
                num_sigmas = len(self.sigma)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                heatmap_size = cfg['heatmap_size']

                target = np.empty(
                    (0, num_joints, heatmap_size[1], heatmap_size[0]),
                    dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_sigmas):
                    target_i, target_weight_i = self._msra_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.sigma[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._msra_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.sigma)

        elif self.encoding == 'Megvii':
            if isinstance(self.kernel, list):
                num_kernels = len(self.kernel)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, num_joints, H, W), dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_kernels):
                    target_i, target_weight_i = self._megvii_generate_target(
                        cfg, joints_3d, joints_3d_visible, self.kernel[i])
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._megvii_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible,
                    self.kernel)

        elif self.encoding == 'UDP':
            if self.target_type.lower() == 'CombinedTarget'.lower():
                factors = self.valid_radius_factor
                channel_factor = 3
            elif self.target_type.lower() == 'GaussianHeatmap'.lower():
                factors = self.sigma
                channel_factor = 1
            else:
                raise ValueError('target_type should be either '
                                 "'GaussianHeatmap' or 'CombinedTarget'")
            if isinstance(factors, list):
                num_factors = len(factors)
                cfg = results['ann_info']
                num_joints = cfg['num_joints']
                W, H = cfg['heatmap_size']

                target = np.empty((0, channel_factor * num_joints, H, W),
                                  dtype=np.float32)
                target_weight = np.empty((0, num_joints, 1), dtype=np.float32)
                for i in range(num_factors):
                    target_i, target_weight_i = self._udp_generate_target(
                        cfg, joints_3d, joints_3d_visible, factors[i],
                        self.target_type)
                    target = np.concatenate([target, target_i[None]], axis=0)
                    target_weight = np.concatenate(
                        [target_weight, target_weight_i[None]], axis=0)
            else:
                target, target_weight = self._udp_generate_target(
                    results['ann_info'], joints_3d, joints_3d_visible, factors,
                    self.target_type)
        else:
            raise ValueError(
                f'Encoding approach {self.encoding} is not supported!')

        results['target'] = target
        results['target_weight'] = target_weight

        return results


class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in `keys` as it is, and collect items in `meta_keys`
    into a meta item called `meta_name`.This is usually the last stage of the
    data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str|tuple]): Required keys to be collected. If a tuple
          (key, key_new) is given as an element, the item retrived by key will
          be renamed as key_new in collected data.
        meta_name (str): The name of the key that contains meta infomation.
          This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str|tuple]): Keys that are collected under
          meta_name. The contents of the `meta_name` dictionary depends
          on `meta_keys`.
    """

    def __init__(self, keys, meta_keys, meta_name='img_metas'):
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        """Performs the Collect formating.

        Args:
            results (dict): The resulting dict to be modified and passed
              to the next transform in pipeline.
        """
        if 'ann_info' in results:
            results.update(results['ann_info'])

        data = {}
        for key in self.keys:
            if isinstance(key, tuple):
                assert len(key) == 2
                key_src, key_tgt = key[:2]
            else:
                key_src = key_tgt = key
            data[key_tgt] = results[key_src]

        meta = {}
        if len(self.meta_keys) != 0:
            for key in self.meta_keys:
                if isinstance(key, tuple):
                    assert len(key) == 2
                    key_src, key_tgt = key[:2]
                else:
                    key_src = key_tgt = key
                meta[key_tgt] = results[key_src]
        if 'bbox_id' in results:
            meta['bbox_id'] = results['bbox_id']
        data[self.meta_name] = meta

        return data

    def __repr__(self):
        """Compute the string representation."""
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')


class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]): Either config
          dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data
