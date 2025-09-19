#!/bin/bash

# 设置 LULA_LIB_DIR
LULA_LIB_DIR="/home/hiyio/anaconda3/envs/env_isaaclab/lib/python3.10/site-packages/isaacsim/exts/isaacsim.robot_motion.lula/pip_prebundle/_lula_libs"

# 更新 LD_LIBRARY_PATH
export LD_LIBRARY_PATH="${LULA_LIB_DIR}:${LD_LIBRARY_PATH}"

# 打印结果确认
echo "LD_LIBRARY_PATH 已更新为: $LD_LIBRARY_PATH"

# 在这里可以加上你要执行的命令
# 例如：python your_script.py
