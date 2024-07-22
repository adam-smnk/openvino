# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest



class TestMlirBinaryOps(PytorchLayerTest):

    def _prepare_input(self):
        return (torch.randint(0, 10, self.lhs_shape).to(self.lhs_type).numpy(),
                torch.randint(0, 10, self.rhs_shape).to(self.rhs_type).numpy())

    def create_model(self, lhs_type, rhs_type):
        class mlir_binary_ops(torch.nn.Module):
            def __init__(self, lhs_type, rhs_type):
                super().__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type

            def forward(self, lhs, rhs):
                add = torch.add(lhs.to(self.lhs_type), rhs.to(self.rhs_type), alpha=2)
                sub = torch.sub(add, rhs.to(self.rhs_type), alpha=0.5)
                mul = torch.mul(sub, lhs.to(self.lhs_type))
                return torch.div(mul, add)

        ref_net = None

        return mlir_binary_ops(lhs_type, rhs_type), ref_net, None

    @pytest.mark.parametrize(("lhs_type", "rhs_type"), [[torch.float32, torch.float32]])
    @pytest.mark.parametrize(("lhs_shape", "rhs_shape"), [([2, 3], [2, 3])])

    def test_mlir_binary(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        # TODO: test with static shapes for XSMM acceleration
        self._test(*self.create_model(lhs_type, rhs_type),
                   ie_device, precision, ir_version, dynamic_shapes=True)
