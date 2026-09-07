#!/bin/bash
set -e

export PATH="/root/.local/bin:/root/miniforge3/bin:$PATH"

cd /path/to/Atom-0

mkdir -p /root/.cache
rm -rf /root/.cache/huggingface /root/.cache/openpi /root/.cache/uv
ln -s /mnt/data/cache/huggingface /root/.cache/huggingface
ln -s /mnt/data/cache/openpi      /root/.cache/openpi
ln -s /mnt/data/cache/uv          /root/.cache/uv

# 启动 server
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=checkpoints/pi05_libero/libero_full_v4/18000 \
  > data/libero/server.log 2>&1 &
SERVER_PID=$!

until grep -q "Creating server" data/libero/server.log 2>/dev/null; do
  sleep 5
done
echo "[INFO] Server ready."

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# 依次跑四个 suite（server 不需要重启）
for SUITE in libero_object; do
  echo "[INFO] Running $SUITE ..."
  mkdir -p data/libero/videos/v4_${SUITE}
  MUJOCO_GL=glx xvfb-run -s "-screen 0 1024x768x24" \
  python examples/libero/main.py \
    --args.task-suite-name ${SUITE} \
    --args.num-trials-per-task 50 \
    --args.video-out-path data/libero/videos/v4_${SUITE} \
    2>&1 | tee data/libero/v4_${SUITE}.log
  echo "[INFO] $SUITE done."
done

kill $SERVER_PID
echo "[INFO] All suites completed."

