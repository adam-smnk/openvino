#!/bin/bash
#
# Runs MLP benchmarks

die_syntax() {
  echo "Syntax: $0 [-t (f32|f16|bf16|...)] [-b (mlp)] [-D]"
  echo ""
  echo "  -t: Optional data type"
  echo "  -b: Optional baseline model"
  echo "  -D: Set model shapes to dynamic"
  exit 1
}

# Cmd-line opts
while getopts "t:b:D" arg; do
  case ${arg} in
    t)
      DATA_TYPE=${OPTARG}
      ;;
    b)
      BASELINE_MODEL=${OPTARG}
      ;;
    D)
      IS_DYNAMIC=true
      ;;
    ?)
      echo "Invalid option: ${OPTARG}"
      die_syntax
      ;;
  esac
done

OV_ROOT=$(git rev-parse --show-toplevel)
BENCH_ROOT=$(realpath ${OV_ROOT}/tools/mlir_bench)

MODEL_GEN=$(realpath ${BENCH_ROOT}/ov_model_gen.py)
BENCH_RUNNER=benchmark_app

# Initial validation.
if ! [ -d ${OV_ROOT} ]; then
  echo "Missing OV repo"
  exit 1
fi
if ! [ -d ${BENCH_ROOT} ]; then
  echo "Missing MLIR benchmark directory"
  exit 1
fi
if ! [ -f ${MODEL_GEN} ]; then
  echo "Missing model generator"
  exit 1
fi
if ! [ "$(command -v ${BENCH_RUNNER})" ]; then
  echo "Missing benchmark runner ${BENCH_RUNNER}"
  exit 1
fi
if [ "${BASELINE_MODEL}" ] && [ ${IS_DYNAMIC} ]; then
  echo "Baseline models with dynamic shapes not supported"
  exit 1
fi

# Kernel config.
INPUT_SIZES=( 1024 2048 4096 8192 )
OUTPUT_SIZES=( 128 256 512 )
if [ ! "${DATA_TYPE}" ]; then
    DATA_TYPE="f32"
fi
MODEL_NAME="MLIR_MLP_BENCH.xml"

for OUT_SIZE in "${OUTPUT_SIZES[@]}"; do
  echo "MLP - OUT: ${OUT_SIZE} INS: ${INPUT_SIZES[@]}"
  for IN_SIZE in "${INPUT_SIZES[@]}"; do
    # Generate model.
    if [ "${BASELINE_MODEL}" ]; then
        # Enable baseline model flag.
        MODEL_CONFIG=(-b="${BASELINE_MODEL}[${OUT_SIZE},${OUT_SIZE},${IN_SIZE}]")
    else
        # Generate default PyTorch MLP.
        MODEL_CONFIG=(-l="linear[${IN_SIZE},${OUT_SIZE}] relu[]")
    fi
    GEN_FLAGS=(-t ${DATA_TYPE} -n ${MODEL_NAME})
    if [ ${IS_DYNAMIC} ]; then
        GEN_FLAGS+=(--dynamic)
    fi
    python3 ${MODEL_GEN} "${MODEL_CONFIG[@]}" "${GEN_FLAGS[@]}"
    if [ $? != 0 ]; then
        echo "Failed to generate model"
        exit 1
    fi
    # Run benchmark.
    PRECISION=${DATA_TYPE}
    if [ "${DATA_TYPE}" = "bf16" ]; then
        # No native support for bf16, use simple f16 instead.
        PRECISION="f16"
    fi
    if [ ${IS_DYNAMIC} ]; then
        DATA_SHAPE=(-data_shape [${OUT_SIZE},${IN_SIZE}])
    fi
    # Benchmark config. Disable parallelism.
    PERF_FLAGS="-niter 10000 -hint none -nstreams 1 -nthreads 1"
    BENCH_FLAGS="-m ${MODEL_NAME} -d CPU \
        -ip ${PRECISION} ${DATA_SHAPE[@]} ${PERF_FLAGS}"
    ${BENCH_RUNNER} ${BENCH_FLAGS} 2>/dev/null | \
        sed -nE "s/.*\[ INFO \]\s*Median:\s*([0-9.]+).*/\\1/p"
  done
done
