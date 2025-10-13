# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from abc import ABCMeta
import copy
import math
import torch.utils.checkpoint as cp
import torchvision
from typing import Optional, Tuple, Union
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single
from .registry import build_conv_layer, CONV_LAYERS, build_norm_layer

# from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.utils import deprecated_api_warning
from torch.autograd import Function
from mmcv.utils import ext_loader, print_log

ext_module = ext_loader.load_ext(
    "_ext", ["modulated_deform_conv_forward", "modulated_deform_conv_backward"]
)


class ModulatedDeformConv2dFunction(Function):
    @staticmethod
    def symbolic(
        g,
        input,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
    ):
        input_tensors = [input, offset, mask, weight]
        if bias is not None:
            input_tensors.append(bias)
        return g.op(
            "mmcv::MMCVModulatedDeformConv2d",
            *input_tensors,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            groups_i=groups,
            deform_groups_i=deform_groups,
        )

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
    ) -> torch.Tensor:
        if input is not None and input.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead."
            )
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)  # type: ignore
        ctx.save_for_backward(input, offset, mask, weight, bias)
        output = input.new_empty(
            ModulatedDeformConv2dFunction._output_size(ctx, input, weight)
        )
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        ext_module.modulated_deform_conv_forward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            output,
            ctx._bufs[1],
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
        grad_output = grad_output.contiguous()
        ext_module.modulated_deform_conv_backward(
            input,
            weight,
            bias,
            ctx._bufs[0],
            offset,
            mask,
            ctx._bufs[1],
            grad_input,
            grad_weight,
            grad_bias,
            grad_offset,
            grad_mask,
            grad_output,
            kernel_h=weight.size(2),
            kernel_w=weight.size(3),
            stride_h=ctx.stride[0],
            stride_w=ctx.stride[1],
            pad_h=ctx.padding[0],
            pad_w=ctx.padding[1],
            dilation_h=ctx.dilation[0],
            dilation_w=ctx.dilation[1],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            with_bias=ctx.with_bias,
        )
        if not ctx.with_bias:
            grad_bias = None

        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be "
                + "x".join(map(str, output_size))
                + ")"
            )
        return output_size


modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply


class ModulatedDeformConv2d(nn.Module):
    @deprecated_api_warning(
        {"deformable_groups": "deform_groups"}, cls_name="ModulatedDeformConv2d"
    )
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: Union[bool, str] = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )


@CONV_LAYERS.register_module("DCNv2")
class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )
        self.init_weights()

    def init_weights(self) -> None:
        super().init_weights()
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, ModulatedDeformConvPack
            # loads previous benchmark models.
            if (
                prefix + "conv_offset.weight" not in state_dict
                and prefix[:-1] + "_offset.weight" in state_dict
            ):
                state_dict[prefix + "conv_offset.weight"] = state_dict.pop(
                    prefix[:-1] + "_offset.weight"
                )
            if (
                prefix + "conv_offset.bias" not in state_dict
                and prefix[:-1] + "_offset.bias" in state_dict
            ):
                state_dict[prefix + "conv_offset.bias"] = state_dict.pop(
                    prefix[:-1] + "_offset.bias"
                )

        if version is not None and version > 1:
            print_log(
                f'ModulatedDeformConvPack {prefix.rstrip(".")} is upgraded to '
                "version 2.",
                logger="root",
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class DCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCN, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.deformable_groups = deformable_groups

        # Generate offset + mask (for modulated DCN)
        offset_mask_channels = 3 * deformable_groups * kernel_size[0] * kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            offset_mask_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        self.init_offset()

        # Deformable convolution weight and bias
        self.weight = nn.Parameter(
            torch.Tensor(
                out_channels,
                in_channels // deformable_groups,
                *kernel_size,
            )
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)

    def init_offset(self):
        nn.init.constant_(self.conv_offset_mask.weight, 0)
        nn.init.constant_(self.conv_offset_mask.bias, 0)

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)

        # Split into offset and mask
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)  # Ensure mask values in [0, 1]

        out = torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return out


# def build_conv_layer(cfg, *args, **kwargs) -> nn.Module:
#     if cfg is None:
#         cfg_ = dict(type="Conv2d", bias=False)
#     else:
#         cfg_ = cfg.copy()
#     layer_type = cfg_.pop("type", "Conv2d")

#     def _pull_io(_args, _kwargs):
#         in_c = _kwargs.pop("in_channels", _args[0] if len(_args) > 0 else None)
#         out_c = _kwargs.pop("out_channels", _args[1] if len(_args) > 1 else None)
#         ksize = _kwargs.pop("kernel_size", _args[2] if len(_args) > 2 else None)
#         if in_c is None or out_c is None or ksize is None:
#             raise ValueError(
#                 "build_conv_layer requires in_channels, out_channels, and kernel_size (positional or keyword)."
#             )
#         return in_c, out_c, ksize

#     if layer_type in ("Conv2d", "Conv"):
#         in_channels, out_channels, kernel_size = _pull_io(args, kwargs)
#         stride = kwargs.pop("stride", 1)
#         padding = kwargs.pop("padding", 0)
#         dilation = kwargs.pop("dilation", 1)
#         groups = kwargs.pop("groups", cfg_.pop("groups", 1))
#         bias = kwargs.pop("bias", cfg_.pop("bias", False))
#         padding_mode = kwargs.pop("padding_mode", cfg_.pop("padding_mode", "zeros"))
#         return nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#             padding_mode=padding_mode,
#         )
#     elif layer_type in ("DCN", "DCNv2"):
#         in_channels, out_channels, kernel_size = _pull_io(args, kwargs)
#         stride = kwargs.pop("stride", 1)
#         padding = kwargs.pop("padding", 0)
#         dilation = kwargs.pop("dilation", 1)
#         # Remove unsupported args if provided
#         kwargs.pop("groups", None)
#         cfg_.pop("groups", None)
#         kwargs.pop("bias", None)
#         cfg_.pop("bias", None)
#         # Support both keys
#         deformable_groups = cfg_.pop(
#             "deform_groups", cfg_.pop("deformable_groups", 1)
#         )

#         # Ensure kernel_size is a tuple[int, int]
#         if isinstance(kernel_size, int):
#             kernel_size_tuple = (kernel_size, kernel_size)
#         else:
#             kernel_size_tuple = _pair(kernel_size)

#         return DCN(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size_tuple,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             deformable_groups=deformable_groups,
#         )
#     else:
#         raise KeyError(f"Unrecognized conv layer type: {layer_type}")


class Linear(torch.nn.Linear):
    def forward(self, x):
        # empty tensor forward of Linear layer is supported in Pytorch 1.6
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 5)):
            out_shape = [x.shape[0], self.out_features]
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameter initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly adds three attributes.

        - ``init_cfg``: the config to control the initialization.
        - ``init_weights``: The function of parameter
            initialization and recording initialization
            information.
        - ``_params_init_info``: Used to track the parameter
            initialization information. This attribute only
            exists during executing the ``init_weights``.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weights() function
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, "_params_init_info"):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is the same before and "
                    f"after calling `init_weights` "
                    f"of {self.__class__.__name__} "
                )
                self._params_init_info[param]["tmp_mean_value"] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else "mmcv"

        from ..cnn import initialize
        from ..cnn.utils.weight_init import update_init_info

        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f"initialize {module_name} with init_cfg {self.init_cfg}",
                    logger=logger_name,
                )
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg["type"] == "Pretrained":
                        return

            for m in self.children():
                if hasattr(m, "init_weights"):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f"Initialized by "
                        f"user-defined `init_weights`"
                        f" in {m.__class__.__name__} ",
                    )

            self._is_init = True
        else:
            warnings.warn(
                f"init_weights of {self.__class__.__name__} has "
                f"been called more than once."
            )

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    # @master_only
    def _dump_init_info(self, logger_name):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """

        logger = get_logger(logger_name)

        with_file_handler = False
        # dump the information to the logger file if there is a `FileHandler`
        for handler in logger.handlers:
            if isinstance(handler, FileHandler):
                handler.stream.write("Name of parameter - Initialization information\n")
                for name, param in self.named_parameters():
                    handler.stream.write(
                        f"\n{name} - {param.shape}: "
                        f"\n{self._params_init_info[param]['init_info']} \n"
                    )
                handler.stream.flush()
                with_file_handler = True
        if not with_file_handler:
            for name, param in self.named_parameters():
                print_log(
                    f"\n{name} - {param.shape}: "
                    f"\n{self._params_init_info[param]['init_info']} \n ",
                    logger=logger_name,
                )

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
            s += f"\ninit_cfg={self.init_cfg}"
        return s


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList in openmmlab.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


class Sequential(BaseModule, nn.Sequential):
    """Sequential module in openmmlab.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


# def build_norm_layer(cfg, num_features, postfix=None):
#     cfg = cfg.copy()
#     layer_type = cfg.pop("type")
#     requires_grad = cfg.pop("requires_grad", None)
#     # Build module
#     if layer_type in ("LN", "LayerNorm"):
#         eps = cfg.pop("eps", 1e-5)
#         elementwise_affine = cfg.pop("elementwise_affine", True)
#         module = nn.LayerNorm(
#             num_features, eps=eps, elementwise_affine=elementwise_affine
#         )
#         abbr = "ln"
#     elif layer_type in ("BN", "BN2d"):
#         module = nn.BatchNorm2d(num_features, **cfg)
#         abbr = "bn"
#     elif layer_type in ("SyncBN", "SyncBatchNorm"):
#         module = nn.SyncBatchNorm(num_features, **cfg)
#         abbr = "syncbn"
#     else:
#         raise KeyError(layer_type)
#     if requires_grad is not None:
#         for p in module.parameters():
#             p.requires_grad = requires_grad
#     if postfix is None:
#         return module
#     name = f"{abbr}{postfix}"
#     return name, module


def build_plugin_layer_hardcoded(cfg, in_channels=None, postfix=""):
    cfg = cfg.copy()
    assert isinstance(postfix, (int, str))
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    layer_type = cfg.pop("type")

    def camel2snack(word):
        import re

        word = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", word)
        word = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", word)
        word = word.replace("-", "_")
        return word.lower()

    abbr = camel2snack(layer_type)

    # Minimal hardcoded support. Default to Identity if unknown.
    layer = nn.Identity()

    name = f"{abbr}{postfix}"
    return name, layer


class Sequential(BaseModule, nn.Sequential):
    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ResLayer(Sequential):
    def __init__(
        self,
        block,
        inplanes,
        planes,
        num_blocks,
        stride=1,
        avg_down=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        downsample_first=True,
        **kwargs,
    ):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            norm_layer = build_norm_layer(norm_cfg, planes * block.expansion)
            if isinstance(norm_layer, tuple):
                norm_layer = norm_layer[1]
            downsample.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=False,
                    ),
                    norm_layer,
                ]
            )
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                )
            )
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs,
                    )
                )
        super(ResLayer, self).__init__(*layers)


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, "Not implemented yet."
        assert plugins is None, "Not implemented yet."

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


# class Bottleneck(BaseModule):
#     expansion = 4

#     def __init__(
#         self,
#         inplanes,
#         planes,
#         stride=1,
#         dilation=1,
#         downsample=None,
#         style="pytorch",
#         with_cp=False,
#         conv_cfg=None,
#         norm_cfg=dict(type="BN"),
#         dcn=None,
#         plugins=None,
#         init_cfg=None,
#     ):
#         super(Bottleneck, self).__init__(init_cfg)
#         assert style in ["pytorch", "caffe"]
#         assert dcn is None or isinstance(dcn, dict)
#         assert plugins is None or isinstance(plugins, list)
#         if plugins is not None:
#             allowed_position = ["after_conv1", "after_conv2", "after_conv3"]
#             assert all(p["position"] in allowed_position for p in plugins)

#         self.inplanes = inplanes
#         self.planes = planes
#         self.stride = stride
#         self.dilation = dilation
#         self.style = style
#         self.with_cp = with_cp
#         self.conv_cfg = conv_cfg
#         self.norm_cfg = norm_cfg
#         self.dcn = dcn
#         self.with_dcn = dcn is not None
#         self.plugins = plugins
#         self.with_plugins = plugins is not None

#         if self.style == "pytorch":
#             self.conv1_stride = 1
#             self.conv2_stride = stride
#         else:
#             self.conv1_stride = stride
#             self.conv2_stride = 1

#         self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
#         self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
#         self.norm3_name, norm3 = build_norm_layer(
#             norm_cfg, planes * self.expansion, postfix=3
#         )

#         self.conv1 = nn.Conv2d(
#             inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False
#         )
#         self.add_module(self.norm1_name, norm1)
#         fallback_on_stride = False
#         if not self.with_dcn or fallback_on_stride:
#             self.conv2 = nn.Conv2d(
#                 planes,
#                 planes,
#                 kernel_size=3,
#                 stride=self.conv2_stride,
#                 padding=dilation,
#                 dilation=dilation,
#                 bias=False,
#             )

#         self.add_module(self.norm2_name, norm2)
#         self.conv3 = nn.Conv2d(
#             planes, planes * self.expansion, kernel_size=1, bias=False
#         )
#         self.add_module(self.norm3_name, norm3)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample

#     def forward_plugin(self, x, plugin_names):
#         out = x
#         for name in plugin_names:
#             out = getattr(self, name)(x)
#         return out

#     @property
#     def norm1(self):
#         """nn.Module: normalization layer after the first convolution layer"""
#         return getattr(self, self.norm1_name)

#     @property
#     def norm2(self):
#         """nn.Module: normalization layer after the second convolution layer"""
#         return getattr(self, self.norm2_name)

#     @property
#     def norm3(self):
#         """nn.Module: normalization layer after the third convolution layer"""
#         return getattr(self, self.norm3_name)

#     def forward(self, x):
#         """Forward function."""

#         def _inner_forward(x):
#             identity = x
#             out = self.conv1(x)
#             out = self.norm1(out)
#             out = self.relu(out)

#             if self.with_plugins:
#                 out = self.forward_plugin(out, self.after_conv1_plugin_names)

#             out = self.conv2(out)
#             out = self.norm2(out)
#             out = self.relu(out)

#             if self.with_plugins:
#                 out = self.forward_plugin(out, self.after_conv2_plugin_names)

#             out = self.conv3(out)
#             out = self.norm3(out)

#             if self.with_plugins:
#                 out = self.forward_plugin(out, self.after_conv3_plugin_names)

#             if self.downsample is not None:
#                 identity = self.downsample(x)

#             out += identity

#             return out

#         if self.with_cp and x.requires_grad:
#             out = cp.checkpoint(_inner_forward, x)
#         else:
#             out = _inner_forward(x)

#         out = self.relu(out)

#         return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ["pytorch", "caffe"]
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ["after_conv1", "after_conv2", "after_conv3"]
            assert all(p["position"] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin["cfg"]
                for plugin in plugins
                if plugin["position"] == "after_conv1"
            ]
            self.after_conv2_plugins = [
                plugin["cfg"]
                for plugin in plugins
                if plugin["position"] == "after_conv2"
            ]
            self.after_conv3_plugins = [
                plugin["cfg"]
                for plugin in plugins
                if plugin["position"] == "after_conv3"
            ]

        if self.style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3
        )

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False
        )
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop("fallback_on_stride", False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        else:
            assert self.conv_cfg is None, "conv_cfg must be None for DCN"
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins
            )
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins
            )
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins
            )

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin, in_channels=in_channels, postfix=plugin.pop("postfix", "")
            )
            assert not hasattr(self, name), f"duplicate plugin {name}"
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResNet(BaseModule):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    ):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")

        block_init_cfg = None
        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be setting at the same time"
        if init_cfg is None:
            self.init_cfg = [
                dict(type="Kaiming", layer="Conv2d"),
                dict(type="Constant", val=1, layer=["_BatchNorm", "GroupNorm"]),
            ]
            block = self.arch_settings[depth][0]
            if self.zero_init_residual:
                if block is BasicBlock:
                    block_init_cfg = dict(
                        type="Constant", val=0, override=dict(name="norm2")
                    )
                elif block is Bottleneck:
                    block_init_cfg = dict(
                        type="Constant", val=0, override=dict(name="norm3")
                    )
        else:
            raise TypeError("pretrained must be a str or None")

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

    def make_stage_plugins(self, plugins, stage_idx):
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop("stages", None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = nn.Conv2d(
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1
        )
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def build_norm_layer(self, cfg, num_features, postfix=None):
        return build_norm_layer(cfg, num_features, postfix)
