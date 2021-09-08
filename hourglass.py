import os
import copy
from abc import ABCMeta, abstractmethod

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from top_down_eval import keypoints_from_heatmaps, pose_pck_accuracy
from post_transforms import flip_back
from mse_loss import JointsMSELoss


def normal_init(module, mean=0.0, std=1.0, bias=0.0):
    if hasattr(module, 'weight') and module.weight is not None:
        initializer = nn.initializer.Normal(mean=mean, std=std)
        initializer(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        initializer = nn.initializer.Constant(value=bias)
        initializer(module.bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        initializer = nn.initializer.Constant(value=val)
        initializer(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        initializer = nn.initializer.Constant(value=bias)
        initializer(module.bias)


def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm2D instead"""
    if paddle.get_device() == 'cpu' or os.environ.get('PADDLESEG_EXPORT_STAGE'):
        return nn.BatchNorm2D(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNLayer(nn.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            is_vd_mode=False,
            act=None,
    ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        if is_vd_mode:
            self._pool2d_avg = nn.AvgPool2D(
                kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 if dilation == 1 else 0,
            dilation=dilation,
            groups=groups,
            bias_attr=False)

        self._batch_norm = SyncBatchNorm(out_channels, momentum=0.1)
        self.act = act
        if act is not None:
            self._act_op = nn.ReLU()

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act is not None:
            y = self._act_op(y)

        return y

class BasicBlock(nn.Layer):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        style (str): `pytorch` or `caffe`. It is unused and reserved for
            unified API with Bottleneck.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu'
        )

        self.conv2 = ConvBNLayer(
            in_channels=self.mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None
        )
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out

        out = _inner_forward(x)

        out = self.relu(out)
        return out


def get_expansion(block, expansion=None):
    """Get the expansion of a residual block.

    The block expansion will be obtained by the following order:

    1. If ``expansion`` is given, just return it.
    2. If ``block`` has the attribute ``expansion``, then return
       ``block.expansion``.
    3. Return the default value according the the block type:
       1 for ``BasicBlock`` and 4 for ``Bottleneck``.

    Args:
        block (class): The block class.
        expansion (int | None): The given expansion ratio.

    Returns:
        int: The expansion of the block.
    """
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion

class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): Residual block used to build ResLayer.
        num_blocks (int): Number of blocks.
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int, optional): The expansion for BasicBlock/Bottleneck.
            If not specified, it will firstly be obtained via
            ``block.expansion``. If the block has no attribute "expansion",
            the following default values will be used: 1 for BasicBlock and
            4 for Bottleneck. Default: None.
        stride (int): stride of the first block. Default: 1.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2D(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True))
            downsample.extend([
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=conv_stride
                )
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            in_channels = out_channels
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        expansion=self.expansion,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
        else:  # downsample_first=False is for HourglassModule
            for i in range(0, num_blocks - 1):
                layers.append(
                    block(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        expansion=self.expansion,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))

        super().__init__(*layers)


class HourglassModule(nn.Layer):
    """Hourglass Module for HourglassNet backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule.
        stage_channels (list[int]): Feature channels of sub-modules in current
            and follow-up HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in current and
            follow-up HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 depth,
                 stage_channels,
                 stage_blocks,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        self.depth = depth

        cur_block = stage_blocks[0]
        next_block = stage_blocks[1]

        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]

        self.up1 = ResLayer(
            BasicBlock, cur_block, cur_channel, cur_channel, norm_cfg=norm_cfg)

        self.low1 = ResLayer(
            BasicBlock,
            cur_block,
            cur_channel,
            next_channel,
            stride=2,
            norm_cfg=norm_cfg)

        if self.depth > 1:
            self.low2 = HourglassModule(depth - 1, stage_channels[1:],
                                        stage_blocks[1:])
        else:
            self.low2 = ResLayer(
                BasicBlock,
                next_block,
                next_channel,
                next_channel,
                norm_cfg=norm_cfg)

        self.low3 = ResLayer(
            BasicBlock,
            cur_block,
            next_channel,
            cur_channel,
            norm_cfg=norm_cfg,
            downsample_first=False)

        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        """Model forward function."""
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class HourglassNet(nn.Layer):
    """HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    More details can be found in the `paper
    <https://arxiv.org/abs/1603.06937>`__ .

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule modules stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channel (int): Feature channel of conv after a HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
         self = HourglassNet()
         self.eval()
         inputs = torch.rand(1, 3, 511, 511)
         level_outputs = self.forward(inputs)
         for level_output in level_outputs:
            print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 downsample_times=5,
                 num_stacks=2,
                 stage_channels=(256, 256, 384, 384, 384, 512),
                 stage_blocks=(2, 2, 2, 2, 2, 4),
                 feat_channel=256,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) == len(stage_blocks)
        assert len(stage_channels) > downsample_times

        cur_channel = stage_channels[0]

        self.stem = nn.Sequential(
            ConvBNLayer(in_channels=3, out_channels=128, kernel_size=7, stride=2,act='relu'),
            ResLayer(BasicBlock, 1, 128, 256, stride=2, norm_cfg=norm_cfg))

        self.hourglass_modules = nn.LayerList([
            HourglassModule(downsample_times, stage_channels, stage_blocks)
            for _ in range(num_stacks)
        ])

        self.inters = ResLayer(
            BasicBlock,
            num_stacks - 1,
            cur_channel,
            cur_channel,
            norm_cfg=norm_cfg)

        self.conv1x1s = nn.LayerList([
            ConvBNLayer(in_channels=cur_channel, out_channels=cur_channel, kernel_size=1)
            for _ in range(num_stacks - 1)
        ])

        self.out_convs = nn.LayerList([
            ConvBNLayer(in_channels=cur_channel, out_channels=feat_channel, kernel_size=3, act='relu')
            for _ in range(num_stacks)
        ])

        self.remap_convs = nn.LayerList([
            ConvBNLayer(in_channels=feat_channel, out_channels=cur_channel,
                        kernel_size=1)
            for _ in range(num_stacks - 1)
        ])

        self.relu = nn.ReLU()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            pass
        elif pretrained is None:
            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    initializer = nn.initializer.Normal(std=0.001)
                    initializer(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        initializer = nn.initializer.Constant(value=0)
                        initializer(m.bias)
                elif isinstance(m, (nn.BatchNorm, nn.GroupNorm)):
                    initializer = nn.initializer.Constant(value=1)
                    if hasattr(m, 'weight') and m.weight is not None:
                        initializer(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        initializer(m.bias)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Model forward function."""
        inter_feat = self.stem(x)
        out_feats = []

        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]
            out_conv = self.out_convs[ind]

            hourglass_feat = single_hourglass(inter_feat)
            out_feat = out_conv(hourglass_feat)
            out_feats.append(out_feat)

            if ind < self.num_stacks - 1:
                inter_feat = self.conv1x1s[ind](
                    inter_feat) + self.remap_convs[ind](
                        out_feat)
                inter_feat = self.inters[ind](self.relu(inter_feat))

        return out_feats


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs



class TopdownHeatmapBaseHead(nn.Layer):
    """Base class for top-down heatmap heads.

    All top-down heatmap heads should subclass it.
    All subclass should overwrite:

    Methods:`get_loss`, supporting to calculate loss.
    Methods:`get_accuracy`, supporting to calculate accuracy.
    Methods:`forward`, supporting to forward model.
    Methods:`inference_model`, supporting to inference model.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_loss(self, **kwargs):
        """Gets the loss."""

    @abstractmethod
    def get_accuracy(self, **kwargs):
        """Gets the accuracy."""

    @abstractmethod
    def forward(self, **kwargs):
        """Forward function."""

    @abstractmethod
    def inference_model(self, **kwargs):
        """Inference function."""

    def decode(self, img_metas, output, **kwargs):
        """Decode keypoints from heatmaps.

        Args:
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            output (np.ndarray[N, K, H, W]): model predicted heatmaps.
        """
        batch_size = len(img_metas)

        if 'bbox_id' in img_metas[0]:
            bbox_ids = []
        else:
            bbox_ids = None

        c = np.zeros((batch_size, 2), dtype=np.float32)
        s = np.zeros((batch_size, 2), dtype=np.float32)
        image_paths = []
        score = np.ones(batch_size)
        for i in range(batch_size):
            c[i, :] = img_metas[i]['center']
            s[i, :] = img_metas[i]['scale']
            image_paths.append(img_metas[i]['image_file'])

            if 'bbox_score' in img_metas[i]:
                score[i] = np.array(img_metas[i]['bbox_score']).reshape(-1)
            if bbox_ids is not None:
                bbox_ids.append(img_metas[i]['bbox_id'])

        preds, maxvals = keypoints_from_heatmaps(
            output,
            c,
            s,
            unbiased=self.test_cfg.get('unbiased_decoding', False),
            post_process=self.test_cfg.get('post_process', 'default'),
            kernel=self.test_cfg.get('modulate_kernel', 11),
            valid_radius_factor=self.test_cfg.get('valid_radius_factor',
                                                  0.0546875),
            use_udp=self.test_cfg.get('use_udp', False),
            target_type=self.test_cfg.get('target_type', 'GaussianHeatmap'))

        all_preds = np.zeros((batch_size, preds.shape[1], 3), dtype=np.float32)
        all_boxes = np.zeros((batch_size, 6), dtype=np.float32)
        all_preds[:, :, 0:2] = preds[:, :, 0:2]
        all_preds[:, :, 2:3] = maxvals
        all_boxes[:, 0:2] = c[:, 0:2]
        all_boxes[:, 2:4] = s[:, 0:2]
        all_boxes[:, 4] = np.prod(s * 200.0, axis=1)
        all_boxes[:, 5] = score

        result = {}

        result['preds'] = all_preds
        result['boxes'] = all_boxes
        result['image_paths'] = image_paths
        result['bbox_ids'] = bbox_ids

        return result

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding



class TopdownHeatmapMultiStageHead(TopdownHeatmapBaseHead):
    """Top-down heatmap multi-stage head.

    TopdownHeatmapMultiStageHead is consisted of multiple branches,
    each of which has num_deconv_layers(>=0) number of deconv layers
    and a simple conv2d layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_stages (int): Number of stages.
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        loss_keypoint (dict): Config for keypoint loss. Default: None.
    """

    def __init__(self,
                 in_channels=512,
                 out_channels=17,
                 num_stages=1,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.in_channels = in_channels
        self.num_stages = num_stages
        self.loss = JointsMSELoss(use_target_weight=True)

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        # build multi-stage deconv layers
        self.multi_deconv_layers = nn.LayerList()
        for _ in range(self.num_stages):
            if num_deconv_layers > 0:
                deconv_layers = self._make_deconv_layer(
                    num_deconv_layers,
                    num_deconv_filters,
                    num_deconv_kernels,
                )
            elif num_deconv_layers == 0:
                deconv_layers = Identity()
            else:
                raise ValueError(
                    f'num_deconv_layers ({num_deconv_layers}) should >= 0.')
            self.multi_deconv_layers.append(deconv_layers)

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        # build multi-stage final layers
        self.multi_final_layers = nn.LayerList()
        for i in range(self.num_stages):
            if identity_final_layer:
                final_layer = Identity()
            else:
                final_layer = nn.Conv2D(in_channels=num_deconv_filters[-1]
                                        if num_deconv_layers > 0 else in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding
                                        )
            self.multi_final_layers.append(final_layer)

    def get_loss(self, output, target, target_weight):
        """Calculate top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            num_outputs: O
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]):
                Output heatmaps.
            target (torch.Tensor[NxKxHxW]):
                Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        losses = dict()

        assert isinstance(output, list)
        assert target.dim() == 4 and target_weight.dim() == 3

        if isinstance(self.loss, nn.Sequential):
            assert len(self.loss) == len(output)
        for i in range(len(output)):
            target_i = target
            target_weight_i = target_weight
            if isinstance(self.loss, nn.Sequential):
                loss_func = self.loss[i]
            else:
                loss_func = self.loss
            loss_i = loss_func(output[i], target_i, target_weight_i)
            if 'mse_loss' not in losses:
                losses['mse_loss'] = loss_i
            else:
                losses['mse_loss'] += loss_i

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            batch_size: N
            num_keypoints: K
            heatmaps height: H
            heatmaps weight: W

        Args:
            output (torch.Tensor[NxKxHxW]): Output heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
            target_weight (torch.Tensor[NxKx1]):
                Weights across different joint types.
        """

        accuracy = dict()

        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(
                output[-1].detach().numpy(),
                target.detach().numpy(),
                target_weight.detach().numpy().squeeze(-1) > 0)
            accuracy['acc_pose'] = float(avg_acc)

        return accuracy

    def forward(self, x):
        """Forward function.

        Returns:
            out (list[Tensor]): a list of heatmaps from multiple stages.
        """
        out = []
        assert isinstance(x, list)
        for i in range(self.num_stages):
            y = self.multi_deconv_layers[i](x[i])
            y = self.multi_final_layers[i](y)
            out.append(y)
        return out

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (List[torch.Tensor[NxKxHxW]]): Input features.
            flip_pairs (None | list[tuple()):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)
        assert isinstance(output, list)
        output = output[-1]

        if flip_pairs is not None:
            # perform flip
            output_heatmap = flip_back(
                output.detach().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().numpy()

        return output_heatmap

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2DTranspose(in_channels=self.in_channels,
                                   out_channels=planes,
                                   kernel_size=kernel,
                                   stride=2,
                                   padding=padding,
                                   output_padding=output_padding,
                                   bias_attr=False)
            )
            layers.append(nn.BatchNorm2D(planes))
            layers.append(nn.ReLU())
            self.in_channels = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.multi_deconv_layers.named_sublayers():
            if isinstance(m, nn.Conv2DTranspose):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2D):
                constant_init(m, 1)
        for m in self.multi_final_layers.sublayers():
            if isinstance(m, nn.Conv2D):
                normal_init(m, std=0.001, bias=0)


if __name__ == '__main__':
    self = HourglassNet()
    self.eval()
    inputs = paddle.rand([1, 3, 511, 511])
    level_outputs = self.forward(inputs)
    for level_output in level_outputs:
        print(tuple(level_output.shape))

    head = TopdownHeatmapMultiStageHead(in_channels=256, out_channels=16, num_stages=1, num_deconv_layers=0)

    pass
