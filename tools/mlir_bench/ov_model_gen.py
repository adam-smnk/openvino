#!/usr/bin/python3

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import argparse
import string
import sys

import torch
import torch.nn as nn
import openvino as ov


class TorchAdd(nn.Module):
    def __init__(self, sizes, type=None):
        super().__init__()
        # Generate random data
        self.tensor = torch.empty(*sizes, dtype=type).data.normal_(0, 0.01)
    def forward(self, a):
        return a + self.tensor


class TorchSub(nn.Module):
    def __init__(self, sizes, type=None):
        super().__init__()
        # Generate random data
        self.tensor = torch.empty(*sizes, dtype=type).data.normal_(0, 0.01)
    def forward(self, a):
        return a - self.tensor


class TorchMul(nn.Module):
    def __init__(self, sizes, type=None):
        super().__init__()
        # Generate random data
        self.tensor = torch.empty(*sizes, dtype=type).data.normal_(0, 0.01)
    def forward(self, a):
        return a * self.tensor


class TorchMatmul(nn.Module):
    def __init__(self, sizes_mnk, type=None):
        super().__init__()
        k = sizes_mnk[2]
        n = sizes_mnk[1]
        # Generate random data
        self.weights = torch.empty(k, n, dtype=type).data.normal_(0, 0.01)
    def forward(self, a):
        return torch.matmul(a, self.weights)


class TorchDiv(nn.Module):
    def __init__(self, sizes, type=None):
        super().__init__()
        # Generate random weights
        self.tensor = torch.empty(*sizes, dtype=type).data.normal_(1, 10)
    def forward(self, a):
        return a / self.tensor


class TorchSequential(nn.Module):
    def __init__(self):
        super(TorchSequential, self).__init__()
        self.model = nn.Sequential()
    def forward(self, a):
        return self.model(a)
    def append(self, module: nn.Module):
        self.model.append(module)


def get_torch_type(type: str) -> torch.dtype:
    if type == 'f32':
        return torch.float32
    if type == 'f16':
        return torch.float16
    if type == 'bf16':
        return torch.bfloat16
    assert False, f"Unsupported torch data type {type}"


def get_torch_layer(layer: str, sizes: list[int], type: str) -> nn.Module:
    data_type = get_torch_type(type)
    if layer == 'linear':
        assert len(sizes) == 2, "invalid sizes for linear"
        linear = nn.Linear(*sizes, dtype=data_type)
        # Generate random weights
        linear.weight.data.normal_(0, 0.01)
        linear.bias.data.fill_(0.01)
        return linear
    if layer == 'relu':
        return nn.ReLU()
    if layer == 'add':
        return TorchAdd(sizes, data_type)
    if layer == 'sub':
        return TorchSub(sizes, data_type)
    if layer == 'mul':
        return TorchMul(sizes, data_type)
    if layer == 'div':
        return TorchDiv(sizes, data_type)
    if layer == 'matmul':
        assert len(sizes) == 3, "invalid sizes for mm"
        return TorchMatmul(sizes, data_type)
    assert False, f"Unsupported torch layer type {layer}"


def get_layer_name(layer_desc: str) -> str:
    return layer_desc[0:layer_desc.find('[')]


def get_layer_sizes(layer_desc: str) -> list[int]:
    desc_sizes = layer_desc[layer_desc.find('[')+1:layer_desc.find(']')]
    return [int(size) for size in filter(None, desc_sizes.split(','))]


def parse_layer(layer_desc: str, type: str) -> nn.Module:
    layer = get_layer_name(layer_desc)
    sizes = get_layer_sizes(layer_desc)
    return get_torch_layer(layer, sizes, type)


def get_ov_type(type: str) -> ov.Type:
    if type == 'f32':
        return ov.Type.f32
    if type == 'f16':
        return ov.Type.f16
    if type == 'bf16':
        return ov.Type.bf16
    assert False, f"Unsupported OV data type {type}"


def get_layer_inputs(layer_desc: str, is_dynamic: bool):
    input_sizes = get_layer_sizes(layer_desc)
    if is_dynamic:
        input_sizes = [-1] * len(input_sizes)

    layer = get_layer_name(layer_desc)

    if layer == 'matmul':
        m = input_sizes[0]
        k = input_sizes[2]
        return [[m,k]]

    # Needs to be reversed for nn.Linear, does it apply to other layers too?
    if layer == 'linear':
        input_sizes.reverse()

    return [input_sizes]


def generate_ov_model(layers_desc: str, data_type: str, shape_type: str, file_name: str):
    layers = layers_desc.split()
    torch_seq = TorchSequential()
    for layer in layers:
        module = parse_layer(layer, data_type)
        torch_seq.append(module)

    input_sizes = get_layer_sizes(layers[0])
    if len(input_sizes) == 0:
        print("Invalid input layer sizes")
        sys.exit(1)

    is_dynamic = shape_type == 'dynamic'
    input_shapes = get_layer_inputs(layers[0], is_dynamic)
    ov_type = get_ov_type(data_type)
    inputs = [(ov.PartialShape(shapes), ov_type) for shapes in input_shapes]

    ov_model = ov.convert_model(torch_seq, input=inputs)
    ov.save_model(ov_model, f"{file_name}")


def main():
    parser = argparse.ArgumentParser(
                        prog='OV Model generator',
                        description='Generate PyTorch model and export as OV .xml')
    parser.add_argument('-l', '--layers', type=str.lower,
                        help='Model layers description. For example:\
                                -l="linear[64,32] relu[] linear[32,16] gelu[]"\
                                -l="matmul[128,128,1024] add[128,128] relu[]"\
                                -l="add[8,8] div[8,8]"')
    parser.add_argument('-t', '--type', default='f32', type=str.lower,
                        help='Data type: f32|f16|bf16|...')
    parser.add_argument('-s', '--shapes', default='static', type=str.lower,
                        help='Model shapes type: static|dynamic')
    parser.add_argument('-n', '--name', default='temp.xml',
                        help='Name for exported XML model')
    args = parser.parse_args()

    generate_ov_model(args.layers, args.type, args.shapes, args.name)

    return 0


if __name__ == '__main__':
    sys.exit(main())
