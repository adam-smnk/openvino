#!/usr/bin/env xonsh

# xonsh can be installed with `pip install xonsh`
# xonsh can then be run by invoking `python -m xonsh`
# this script in particular can be invoked with `python -m xonsh lora-benchmark.xsh`

import openvino as ov
from openvino.runtime.op import Constant
from openvino_devtools.builder import OpFactory, outputs_to_nodes
import numpy as np
from pprint import pprint
import re


CONFIGS = [
  [8], [16], [32], [64], [128], [256], [512], [1024]
]
ITERATIONS = 10

BENCH_RUNNER="tpp-run"
RUNNER_FLAGS=f"-entry-point-result=void -e entry -seed 123 -n {ITERATIONS}".split()


def build_lora_model(input_dim=-1, weight_dim=2048, lora_dim=8):
    opset = OpFactory('opset13')

    #t40 = opset.Parameter({'shape': [-1, -1, 2048], 'element_type': 'f32'}, output_names=[{'x'}])  # Input data
    t40 = opset.Parameter({'shape': [input_dim, weight_dim], 'element_type': 'f32'}, output_names=[{'x'}])  # Input data
    t52 = opset.Parameter({'shape': [1, lora_dim], 'element_type': 'f32'}, output_names=[{'alpha'}])  # LoRA alpha parameter

    t48 = Constant(np.random.rand(weight_dim, weight_dim).astype(np.float32))   #  -> f32[2048,2048]  # Original weight matrix W (usually it is compressed to bf16/f16/u8/u4 and represented as a sub-graph)
    t50 = Constant(np.random.rand(lora_dim, weight_dim).astype(np.float32))  #  -> f32[8,2048]   # LoRA matrix A
    t54 = Constant(np.random.rand(weight_dim, lora_dim).astype(np.float32))  #  -> f32[2048,8]   # LoRA matrix B

    t49 = opset.MatMul([t40, t48], {'transpose_a': False, 'transpose_b': True})  # f32[?,?,2048], f32[2048,2048] -> f32[?,?,2048]
    t51 = opset.MatMul([t40, t50], {'transpose_a': False, 'transpose_b': True})  # f32[?,?,2048], f32[8,2048] -> f32[?,?,8]
    t53 = opset.Multiply([t51, t52], {'auto_broadcast': 'numpy'})  # f32[?,?,8], f32[1,8] -> f32[?,?,8]
    t55 = opset.MatMul([t53, t54], {'transpose_a': False, 'transpose_b': True})  # f32[?,?,8], f32[2048,8] -> f32[?,?,2048]
    t56 = opset.Add([t49, t55], {'auto_broadcast': 'numpy'})  # f32[?,?,2048], f32[?,?,2048] -> f32[?,?,2048]
    t57 = opset.Result([t56], {})  # f32[?,?,2048] -> f32[?,?,2048]

    parameters = [t40, t52]
    results = [t57]
    sinks = []
    return ov.Model(outputs_to_nodes(results), outputs_to_nodes(sinks), outputs_to_nodes(parameters))


def main():
    no_mlir_averages = []
    mlir_averages = []
    no_ov_averages = []
    for config in CONFIGS:
        model_desc = '.'.join(str(x) for x in config)
        model_xml = f"lora.{model_desc}.xml"
        model = build_lora_model(*config)
        ov.save_model(model, model_xml)

        BENCH_FLAGS=f"-m {model_xml} -d CPU -ip f32 -infer_precision f32 -hint none -nstreams 1 -nthreads 1".split()

        def do_it(env_str):
            out = $(env @(env_str) benchmark_app @(BENCH_FLAGS) -niter @(ITERATIONS))
            match = re.search(r"Median: +(\d.*) ms", out)
            return float(match.group(1))
        no_mlir_averages.append(do_it("OV_MLIR=0"))
        mlir_averages.append(do_it("OV_MLIR=1"))

        raw_kernel_secs = $(env OV_MLIR=1 OV_MLIR_TPP=1 OV_MLIR_DEBUG=1 benchmark_app @(BENCH_FLAGS) -niter 1 2>&1 | awk '/Source MLIR:/{flag=1; next} /Target LLVM:/{flag=0} flag' | grep -vE '^[-]+$' | tpp-run @(RUNNER_FLAGS))
        no_ov_averages.append(float(raw_kernel_secs) * 1000)

    print("CONFIGS", CONFIGS)
    print("OV NO-MLIR", no_mlir_averages)
    print("OV MLIR", mlir_averages)
    print("NO-OV MLIR", no_ov_averages)


if __name__ == "__main__":
    main()
