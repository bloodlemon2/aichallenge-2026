# ACT LeRobot Workspace

AI Challenge の rosbag を LeRobotDataset に変換し、LeRobot の ACT policy で模倣学習するための作業ディレクトリです。

## 前提

LeRobot は AI Challenge の ROS/Docker 環境ではなく、ホスト側の conda 環境で使います。

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-aichallenge

cd ~/aichallenge/aichallenge-2026/aichallenge/ml_workspace/act_lerobot
```

LeRobot v0.6.0 を使う場合、`~/aichallenge/lerobot` を editable install しておきます。

```bash
cd ~/aichallenge/lerobot
pip install -e ".[training]"
pip install rosbags opencv-python pyyaml tqdm
```

SmolVLAなどのVLAを学習する場合は、追加でVLA用依存関係を入れます。

```bash
cd ~/aichallenge/lerobot
pip install -e ".[smolvla]"
```

確認:

```bash
python -c "import lerobot; print(lerobot.__version__)"
which lerobot-train
```

## 1. 学習用 rosbag 収集

ACTには画像と教師制御が必要です。最低限必要なtopicは以下です。

- `/sensing/camera/image_raw`
- `/control/command/control_cmd`

速度も観測stateに入れる場合は以下も記録します。

- `/vehicle/status/velocity_status`

AWSIM のカメラがOFFだと画像topicは出ません。`aichallenge/simulator_scripts/dev.sh` の `--camera off` を `--camera cpu` または `--camera gpu` にしてから、シミュレータを起動し直してください。

```bash
make down
make dev
```

記録前に画像topicが流れているか確認します。

```bash
ROS_DOMAIN_ID=1 docker compose run --rm --no-deps autoware-command \
  bash -lc 'source /opt/ros/humble/setup.bash && source /aichallenge/workspace/install/setup.bash && ros2 topic hz /sensing/camera/image_raw'
```

記録:

```bash
ROS_DOMAIN_ID=1 docker compose run --rm --no-deps autoware-command \
  bash -lc 'source /opt/ros/humble/setup.bash && source /aichallenge/workspace/install/setup.bash && ros2 bag record /clock /sensing/camera/image_raw /control/command/control_cmd /vehicle/status/velocity_status -s mcap -o /output/d1_learning_bag'
```

終了は記録している端末で `Ctrl+C` です。`metadata.yaml` に以下があれば画像入りbagです。

```text
name: /sensing/camera/image_raw
type: sensor_msgs/msg/Image
```

## 2. LeRobotDataset へ変換

`--state-mode none` は画像のみ、`--state-mode vehicle_status` は速度stateも使います。今のACT実装では `vehicle_status` を推奨します。

```bash
cd ~/aichallenge/aichallenge-2026/aichallenge/ml_workspace/act_lerobot
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-aichallenge

python3 convert_rosbag_to_lerobot.py \
  --bags-dir ~/aichallenge/aichallenge-2026/aichallenge/ml_workspace/dataset/d1_learning_bag \
  --outdir ./dataset/aichallenge_act \
  --repo-id local/aichallenge_act \
  --fps 40 \
  --state-mode vehicle_status \
  --task "drive the racing kart around the course" \
  --overwrite
```

変換結果の確認:

```bash
python -c "from lerobot.datasets import LeRobotDataset; ds=LeRobotDataset('local/aichallenge_act', root='./dataset/aichallenge_act'); print(ds.num_frames, ds.num_episodes); print(ds.features)"
```

期待するfeature:

- `observation.images.front`: RGB画像
- `action`: `[acceleration, steering_tire_angle]`
- `observation.state`: `[longitudinal_velocity, heading_rate]`

VLAでは `--task` の自然言語文が入力として使われます。単一タスクなら固定文で問題ありません。複数タスクを混ぜる場合は、episodeごとに異なるtask文を付けて変換してください。

### データセット名を変えて変換する場合

複数のrosbagや車両ごとのデータを分けて管理する場合は、変換時に `--repo-id` と `--outdir` を変えます。

例: `d2_learning_bag` を別データセットとして作る場合:

```bash
python3 convert_rosbag_to_lerobot.py \
  --bags-dir ~/aichallenge/aichallenge-2026/aichallenge/ml_workspace/dataset/d2_learning_bag \
  --outdir ./dataset/aichallenge_act_d2 \
  --repo-id local/aichallenge_act_d2 \
  --fps 40 \
  --state-mode vehicle_status \
  --overwrite
```

この場合、学習時には以下の2つを同じ組み合わせで指定します。

```text
DATASET_REPO_ID=local/aichallenge_act_d2
DATASET_ROOT=./dataset/aichallenge_act_d2
```

## 3. GPU確認

GPUが見えない場合は、NVIDIAデバイスノードを作成します。

```bash
sudo nvidia-modprobe -u -c=0
nvidia-smi
```

conda環境から確認:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

`True` とGPU名が出れば `DEVICE=cuda` で学習できます。

## 4. ResNet18 重みをネットから取得

ACTのデフォルトbackboneは ResNet18 です。ネット接続できる環境では、事前学習重みを先にキャッシュしておくと学習開始時に止まりません。

```bash
python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"
```

保存先:

```text
~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

ネット接続がない場合は、`PRETRAINED_BACKBONE_WEIGHTS=null` のまま学習します。`train_act.bash` のデフォルトはネット不要の `null` です。

## 5. ACT 学習

学習に使うデータセットは `DATASET_REPO_ID` と `DATASET_ROOT` で指定します。変換時の `--repo-id` と `--outdir` に対応させてください。

デフォルトは以下です。

```text
DATASET_REPO_ID=local/aichallenge_act
DATASET_ROOT=./dataset/aichallenge_act
```

スモークテスト:

```bash
DEVICE=cpu STEPS=1 BATCH_SIZE=1 NUM_WORKERS=0 SAVE_FREQ=1 LOG_FREQ=1 OUTPUT_DIR=./outputs/train/act_smoke ./train_act.bash
```

GPU学習:

```bash
DEVICE=cuda \
STEPS=100000 \
BATCH_SIZE=8 \
NUM_WORKERS=4 \
SAVE_FREQ=5000 \
LOG_FREQ=100 \
OUTPUT_DIR=./outputs/train/act_$(date +%Y%m%d_%H%M%S) \
./train_act.bash
```

別データセットを指定してGPU学習する例:

```bash
DATASET_REPO_ID=local/aichallenge_act_d2 \
DATASET_ROOT=./dataset/aichallenge_act_d2 \
DEVICE=cuda \
STEPS=100000 \
BATCH_SIZE=8 \
NUM_WORKERS=4 \
SAVE_FREQ=5000 \
LOG_FREQ=100 \
OUTPUT_DIR=./outputs/train/act_d2_$(date +%Y%m%d_%H%M%S) \
./train_act.bash
```

ResNet18事前学習重みを使う場合:

```bash
DEVICE=cuda \
PRETRAINED_BACKBONE_WEIGHTS=ResNet18_Weights.IMAGENET1K_V1 \
STEPS=100000 \
BATCH_SIZE=8 \
NUM_WORKERS=4 \
SAVE_FREQ=5000 \
LOG_FREQ=100 \
OUTPUT_DIR=./outputs/train/act_$(date +%Y%m%d_%H%M%S) \
./train_act.bash
```

これはACT全体の学習済みpolicyを読む指定ではなく、画像backboneだけをImageNet重みで初期化する指定です。

### 学習済みACTを別データでfine-tuneする場合

一度自分のデータで学習したACT policyを、新しいデータセットでさらに学習できます。この場合は `--policy.path` に既存checkpointの `pretrained_model` ディレクトリを指定します。

例: `aichallenge_act_d2` で追加学習する場合:

```bash
lerobot-train \
  --dataset.repo_id=local/aichallenge_act_d2 \
  --dataset.root=./dataset/aichallenge_act_d2 \
  --policy.path=./outputs/train/act_20260825_202544/checkpoints/100000/pretrained_model \
  --policy.device=cuda \
  --output_dir=./outputs/train/act_finetune_d2_$(date +%Y%m%d_%H%M%S) \
  --job_name=act_finetune_d2 \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --steps=50000 \
  --batch_size=8 \
  --num_workers=4 \
  --save_freq=5000 \
  --log_freq=100
```

`--policy.path` には以下のようなディレクトリを指定します。

```text
outputs/train/<old_run>/checkpoints/<step>/pretrained_model
```

fine-tune後に走行で使うには、fine-tuneしたcheckpointをONNXへexportし直してください。

```bash
python3 export_act_to_onnx.py \
  --policy-path ./outputs/train/act_finetune_d2_20260828_210000/checkpoints/050000/pretrained_model \
  --output ./ckpt/act_policy.onnx
```

### 中断した学習runを再開する場合

同じrunを中断地点から続ける場合はfine-tuneではなく `--resume=true` を使います。データセットや設定を変えずに、optimizerやschedulerの状態も含めて再開します。

```bash
lerobot-train \
  --resume=true \
  --config_path=./outputs/train/act_20260825_202544/checkpoints/100000/train_config.json
```

使い分け:

```text
画像backboneだけ事前学習重みを使う: PRETRAINED_BACKBONE_WEIGHTS=ResNet18_Weights.IMAGENET1K_V1
新しいデータセットで追加学習する: --policy.path=<pretrained_model>
中断した同じrunを続ける: --resume=true --config_path=<train_config.json>
```

CPUで短く動かす場合:

```bash
DEVICE=cpu STEPS=5000 BATCH_SIZE=1 NUM_WORKERS=0 SAVE_FREQ=1000 LOG_FREQ=50 OUTPUT_DIR=./outputs/train/act_cpu_$(date +%Y%m%d_%H%M%S) ./train_act.bash
```

40Hz走行では `CHUNK_SIZE=10..30` から試すのが無難です。`100` は2.5秒先までのchunkになり、車両制御では長すぎる場合があります。現在のデフォルトは `20` です。

学習済みpolicyは以下のような場所に保存されます。LeRobot v0.6.0では `checkpoints/last` ではなく、ステップ番号のディレクトリを指定します。

```text
outputs/train/<run_name>/checkpoints/100000/pretrained_model
```

## 6. SmolVLA 学習

LeRobot v0.6.0には `smolvla`, `pi0`, `pi05`, `vla_jepa`, `xvla` などのVLA系policyがあります。AI Challengeで最初に試す対象としては、比較的小さい `smolvla` を想定しています。

SmolVLAは画像、state、actionに加えて、LeRobotDataset内の `task` 文字列を使います。データ変換時に `--task` を指定してください。

事前学習済みSmolVLAからfine-tuneする例:

```bash
cd ~/aichallenge/aichallenge-2026/aichallenge/ml_workspace/act_lerobot
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-aichallenge

POLICY_PATH=lerobot/smolvla_base \
DATASET_REPO_ID=local/aichallenge_act \
DATASET_ROOT=./dataset/aichallenge_act \
DEVICE=cuda \
STEPS=50000 \
BATCH_SIZE=1 \
NUM_WORKERS=2 \
SAVE_FREQ=5000 \
LOG_FREQ=100 \
OUTPUT_DIR=./outputs/train/smolvla_$(date +%Y%m%d_%H%M%S) \
./train_smolvla.bash
```

ネットから `lerobot/smolvla_base` を取得できない場合、またはscratchで試す場合:

```bash
POLICY_PATH=scratch \
DATASET_REPO_ID=local/aichallenge_act \
DATASET_ROOT=./dataset/aichallenge_act \
DEVICE=cuda \
STEPS=50000 \
BATCH_SIZE=1 \
NUM_WORKERS=2 \
OUTPUT_DIR=./outputs/train/smolvla_scratch_$(date +%Y%m%d_%H%M%S) \
./train_smolvla.bash
```

SmolVLAはACTより重いので、最初は `BATCH_SIZE=1` から確認してください。メモリに余裕があれば `BATCH_SIZE=2` 以上を試します。

ACTはONNX化してDocker内で推論します。SmolVLAはACTより重く、ROS Humble Docker内へLeRobot v0.6.0を直接入れにくいため、ホスト側condaの推論サーバとDocker内ROS2 bridge nodeに分けます。

```text
実装済み: ホスト側condaでSmolVLA推論サーバを動かし、Docker内ROS2 bridge nodeがHTTPで呼ぶ
要検証: SmolVLAをONNX/TensorRT等へexportしてDocker内で直接推論する
非推奨: ROS Humble Docker内へLeRobot v0.6.0を直接入れる
```

### SmolVLA remote推論で走行する場合

LeRobot v0.6.0はPython 3.12前提、ROS Humbleは通常Python 3.10前提なので、同じPythonプロセスに混ぜません。ホスト側condaでSmolVLA推論HTTPサーバを起動し、Docker内のROS2 bridge nodeがそのサーバを呼びます。

端末1: ホスト側condaでSmolVLA推論サーバを起動します。

```bash
cd ~/aichallenge/aichallenge-2026/aichallenge/ml_workspace/act_lerobot
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-aichallenge

python3 smolvla_inference_server.py \
  --policy-path ./outputs/train/smolvla_20260828_210000/checkpoints/050000/pretrained_model \
  --device cuda \
  --host 127.0.0.1 \
  --port 8765 \
  --task "drive the racing kart around the course"
```

端末2: AI Challengeを `smolvla_remote` で起動します。conda環境に入る必要はありません。

```bash
cd ~/aichallenge/aichallenge-2026
make down
CONTROL_METHOD=smolvla_remote make dev
```

`dev4`で1台だけSmolVLA、他をMPCにする例:

```bash
D1_CONTROL_METHOD=smolvla_remote make dev4
```

SmolVLAで制御しているか確認:

```bash
ROS_DOMAIN_ID=1 docker compose -p 1 run --rm --no-deps autoware-command \
  bash -lc 'source /opt/ros/humble/setup.bash && source /aichallenge/workspace/install/setup.bash && ros2 topic info -v /control/command/control_cmd'
```

`Node name: smolvla_remote_controller_node` が出ればSmolVLA remote制御です。

サーバ疎通確認:

```bash
curl http://127.0.0.1:8765/health
```

SmolVLAはACTより重いため、40Hzで毎フレーム推論できない可能性があります。このbridge nodeはaction chunkを受け取り、次のchunkが必要になった時だけHTTP推論を呼びます。遅い場合は `n_action_steps` を増やす、画像サイズを下げる、またはVLAを低周期で使ってMPC/ACTへ目標を渡す構成を検討してください。

## 7. ONNX export

`make dev` のAutoware Docker内では LeRobot v0.6.0 を直接使いません。学習済みpolicyをONNXへ変換し、ROS2ノードは `onnxruntime` だけで推論します。

```bash
cd ~/aichallenge/aichallenge-2026/aichallenge/ml_workspace/act_lerobot
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-aichallenge

python3 export_act_to_onnx.py \
  --policy-path ./outputs/train/act_20260825_202544/checkpoints/100000/pretrained_model \
  --output ./ckpt/act_policy.onnx
```

ONNXの確認:

```bash
python -c "import onnxruntime as ort, numpy as np; s=ort.InferenceSession('./ckpt/act_policy.onnx', providers=['CPUExecutionProvider']); y=s.run(None, {'image':np.zeros((1,3,200,320),np.float32), 'state':np.zeros((1,2),np.float32)})[0]; print(y.shape, y[0,0])"
```

期待するshape:

```text
(1, 20, 2)
```

## 8. Docker image更新

ROS2推論には `onnxruntime` が必要です。`requirements.txt` に追加済みなので、Docker imageを再ビルドします。

```bash
cd ~/aichallenge/aichallenge-2026
./docker_build.sh dev
make autoware-build
```

コンテナ内で確認:

```bash
docker compose run --rm --no-deps autoware-command \
  bash -lc 'python3 -c "import onnxruntime as ort; print(ort.__version__)"'
```

## 9. ROS2 推論

`CONTROL_METHOD=act_lerobot make dev` で起動する場合、policy pathはDocker内パスです。デフォルトは以下です。

```text
/aichallenge/ml_workspace/act_lerobot/ckpt/act_policy.onnx
```

通常の起動:

```bash
cd ~/aichallenge/aichallenge-2026
make down
CONTROL_METHOD=act_lerobot make dev
```

ACTで制御しているか確認:

```bash
ROS_DOMAIN_ID=1 docker compose run --rm --no-deps autoware-command \
  bash -lc 'source /opt/ros/humble/setup.bash && source /aichallenge/workspace/install/setup.bash && ros2 topic info -v /control/command/control_cmd'
```

`Node name: act_lerobot_controller_node` ならACTです。

ホスト側で単体 `ros2 launch` する場合、AI Challenge workspaceをsourceします。Docker内 `make autoware-build` の symlink install はホストからリンク先が壊れることがあるため、見えない場合は `act_lerobot_controller` だけホストでビルドします。

```bash
source /opt/ros/humble/setup.bash
cd ~/aichallenge/aichallenge-2026/aichallenge/workspace
colcon build --packages-select act_lerobot_controller --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/act_lerobot_controller/share/act_lerobot_controller/local_setup.bash
```

確認:

```bash
ros2 pkg list | grep act_lerobot_controller
```

起動:

```bash
ros2 launch act_lerobot_controller act_lerobot.launch.xml \
  policy_path:=/home/hirosueryoko/aichallenge/aichallenge-2026/aichallenge/ml_workspace/act_lerobot/ckpt/act_policy.onnx \
  use_sim_time:=true
```

`reference.launch.xml` から使う場合:

```bash
ros2 launch aichallenge_submit_launch reference.launch.xml \
  control_method:=act_lerobot \
  simulation:=true \
  use_sim_time:=true
```

## よくあるエラー

`LeRobot is not installed`:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-aichallenge
```

`no synchronized samples; skipped`:

rosbagに `/sensing/camera/image_raw` が入っていません。AWSIMの `--camera off` を直して、画像topicが流れていることを確認してから再記録してください。

`Extra features: {'timestamp'}`:

LeRobot v0.6.0では `timestamp` は自動生成です。このリポジトリの最新版 `convert_rosbag_to_lerobot.py` を使ってください。

`Package 'act_lerobot_controller' not found`:

AI Challenge workspaceがsourceされていません。ホスト起動なら `act_lerobot_controller` をホスト側でビルドし、`local_setup.bash` をsourceしてください。
