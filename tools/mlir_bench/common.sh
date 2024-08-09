#!/bin/bash

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Find the git root directory
git_root() {
  if [ "$(command -v git)" ]; then
    git rev-parse --show-toplevel
  else
    echo "ERROR: missing prerequisites!"
    exit 1
  fi
}

get_ov_build_dir() {
  ROOT=$(git_root)
  BUILD_DIR=$(realpath "${ROOT}/build")
  if [ ! -d "${BUILD_DIR}" ]; then
    echo "ERROR: missing OV build dir!"
    exit 1
  fi
  echo ${BUILD_DIR}
}

grep_ov_cmake_cache() {
  STR=$1

  BUILD=$(get_ov_build_dir)
  CACHE=$(realpath "${BUILD}/CMakeCache.txt")
  if [ ! -f "${CACHE}" ]; then
    echo "ERROR: missing OV CMakeCache!"
    exit 1
  fi

  cat ${CACHE} | grep -i ${STR}
}

get_cache_val() {
  CACHED_STR=$1
  echo ${CACHED_STR} | sed 's/^.*=//'
}

find_openvino_install() {
  if [ "${INTEL_OPENVINO_DIR}" ]; then
    echo "${INTEL_OPENVINO_DIR}"
    exit 0
  fi

  CACHED=$(grep_ov_cmake_cache "CMAKE_INSTALL_PREFIX:PATH")
  INSTALL_DIR=$(get_cache_val ${CACHED})
  if [ ! -d "${INSTALL_DIR}" ]; then
    echo "ERROR: missing OV install dir!"
    exit 1
  fi
  echo "${INSTALL_DIR}"
}

init_openvino() {
  INSTALL_DIR=$(find_openvino_install)
  source ${INSTALL_DIR}/setupvars.sh
}
