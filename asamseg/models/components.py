
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.layers import split_args, Act
from typing import Optional, Sequence, Tuple, Union


def conv_norm_act(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    norm: Optional[Union[Tuple, str]] = 'batch',
    act: Optional[Union[Tuple, str]] = ('relu', {'inplace': True}),
    bias: bool = False,
    dropout: Union[Tuple, str, float, None] = None,
) -> Convolution:
    return Convolution(
        dimensions=2, in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, bias=bias, act=act, norm=norm, dropout=dropout
    )


def conv1x1_norm_act(
    in_channels: int,
    out_channels: int,
    norm: Optional[Union[Tuple, str]] = 'batch',
    act: Optional[Union[Tuple, str]] = ('relu', {'inplace': True}),
    bias: bool = False
) -> Convolution:
    return Convolution(
        dimensions=2, in_channels=in_channels, out_channels=out_channels,
        kernel_size=1, bias=bias, act=act, norm=norm
    )


def conv3x3_norm_act(
    in_channels: int,
    out_channels: int,
    strides: Union[Sequence[int], int] = 1,
    norm: Optional[Union[Tuple, str]] = 'batch',
    act: Optional[Union[Tuple, str]] = ('relu', {'inplace': True}),
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    is_transposed: bool = False,
    padding: Optional[Union[Sequence[int], int]] = None,
    output_padding: Optional[Union[Sequence[int], int]] = None,
) -> Convolution:
    return Convolution(
        dimensions=2, in_channels=in_channels, out_channels=out_channels,
        strides=strides, kernel_size=3, act=act, norm=norm, dropout=dropout, bias=bias,
        is_transposed=is_transposed, padding=padding, output_padding=output_padding
    )


class ResBlock(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = 'NDA',
        norm: Optional[Union[Tuple, str]] = 'batch',
        act: Optional[Union[Tuple, str]] = ('relu', {'inplace': True}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1, # 在Pytorch中，dilation = 1等同于没有dilation的标准卷积
        groups: int = 1,
        bias: bool = False,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        output_padding: Optional[Union[Sequence[int], int]] = None,
        n_layers: int = 2   # 层数
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError( 'ResBlock.__init__(): n_layers can not be less than one!')
        act_name, act_args = split_args(act)
        self.act_func = Act[act_name](**act_args)
        self.adjust_dimension_conv = None
        if in_channels != out_channels:
            self.adjust_dimension_conv = conv1x1_norm_act(
                in_channels, out_channels, norm=norm, act=None
            )

        self.conv= nn.Sequential()
        for i in range(n_layers):
            if i == n_layers - 1:
                act = None
            self.conv.add_module(
                f'layer{i:d}',
                Convolution(
                    dimensions=dimensions, in_channels=in_channels, out_channels=out_channels,
                    strides=strides, kernel_size=kernel_size, adn_ordering=adn_ordering,
                    act=act, norm=norm, dropout=dropout, dropout_dim=dropout_dim,
                    dilation=dilation, groups=groups, bias=bias, conv_only=conv_only,
                    is_transposed=is_transposed, padding=padding, output_padding=output_padding
                )
            )
            in_channels = out_channels


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x)
        if self.adjust_dimension_conv is not None:
            identity = self.adjust_dimension_conv(x)
        out = out + identity
        return self.act_func(out)


class StackedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = 'NDA',
        norm: Optional[Union[Tuple, str]] = 'batch',
        act: Optional[Union[Tuple, str]] = ('relu', {'inplace': True}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = False,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        output_padding: Optional[Union[Sequence[int], int]] = None,
        n_blocks: int = 1,
        use_res_block: bool = False
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError( 'StackedBlock.__init__(): n_blocks can not be less than one!')
        self.conv= nn.Sequential()
        conv_block_type = ResBlock if use_res_block else Convolution
        for i in range(n_blocks):
            self.conv.add_module(
                f'block{i:d}',
                conv_block_type(
                    dimensions=2, in_channels=in_channels, out_channels=out_channels,
                    strides=strides, kernel_size=kernel_size, adn_ordering=adn_ordering,
                    act=act, norm=norm, dropout=dropout, dropout_dim=dropout_dim,
                    dilation=dilation, groups=groups, bias=bias, conv_only=conv_only,
                    is_transposed=is_transposed, padding=padding, output_padding=output_padding
                )
            )
            in_channels = out_channels


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
