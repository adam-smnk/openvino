#!/bin/bash
#
# Runs MLP benchmarks

die_syntax() {
  echo "Syntax: $0 [-t (f32|f16|bf16|...)] [-D]"
  echo ""
  echo "  -t: Optional data type"
  echo "  -D: Set model shapes to dynamic"
  exit 1
}

# Cmd-line opts
while getopts "t:D" arg; do
  case ${arg} in
    t)
      DATA_TYPE=${OPTARG}
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
  echo "Missing benchmark runner"
  exit 1
fi

# Kernel config.
INPUT_SIZES=( 1024 2048 4096 8192 )
OUTPUT_SIZES=( 128 256 512 )
if [ ! "${DATA_TYPE}" ]; then
    DATA_TYPE="f32"
fi
SHAPES="static"
if [ ${IS_DYNAMIC} ]; then
    SHAPES="dynamic"
fi
MODEL_NAME="MLIR_MLP_BENCH.xml"

for OUT_SIZE in "${OUTPUT_SIZES[@]}"; do
  echo "MLP - OUT: ${OUT_SIZE} INS: ${INPUT_SIZES[@]}"
  for IN_SIZE in "${INPUT_SIZES[@]}"; do
    # Generate model.
    MODEL_FLAGS=(-l="linear[${IN_SIZE},${OUT_SIZE}] relu[]"
        -t ${DATA_TYPE} -s ${SHAPES} -n ${MODEL_NAME})
    python3 ${MODEL_GEN} "${MODEL_FLAGS[@]}"
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
    BENCH_FLAGS="-m ${MODEL_NAME} -d CPU -data_shape [${OUT_SIZE,IN_SIZE}]\
        -ip ${PRECISION}"
    ${BENCH_RUNNER} ${BENCH_FLAGS} 2>/dev/null | \
        sed -nE "s/.*\[ INFO \]\s*Median:\s*([0-9.]+).*/\\1/p"
  done
done
