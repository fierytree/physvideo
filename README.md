
## Install
```bash
conda create -n physvideo python=3.9
conda activate physvideo


pip install -r requirements.txt
pip install -v . 
pip install git+https://github.com/hpcaitech/TensorNVMe.git
pip install git+https://github.com/hpcaitech/ColossalAI.git
pip install git+https://github.com/openai/CLIP.git

```

(Optional, recommended for fast speed, especially for training) To enable `layernorm_kernel` and `flash_attn`, you need to install `apex` and `flash-attn` with the following commands.

```bash
# install flash attention
# set enable_flash_attn=False in config to disable flash attention
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install apex
# set enable_layernorm_kernel=False in config to disable apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
```



## Pre-trained Model Weights

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download hpcai-tech/OpenSora-STDiT-v4-360p --local-dir ./ckpts/OpenSora-STDiT-v4-360p
huggingface-cli download hpcai-tech/OpenSora-VAE-v1.3 --local-dir ./ckpts/OpenSora-VAE-v1.3
huggingface-cli download google/t5-v1_1-xxl --local-dir ./ckpts/t5-v1_1-xxl
```

## Training
```bash
python RL.py --epochs 50000 --model_path ./ckpts/OpenSora-STDiT-v4-360p
```

## Inference


```bash
# single prompt
CUDA_VISIBLE_DEVICES=1 python scripts/inference.py configs/opensora-v1-3/inference/t2v.py \
  --num-frames 49 --resolution 360p --aspect-ratio 9:16 \
  --prompt "a beautiful waterfall"

# physvideobench
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2 scripts/inference.py configs/opensora-v1-3/inference/t2v.py \
  --num-frames 49 --resolution 360p --aspect-ratio 9:16 \
  --prompt-path physvideobench.txt

# ucf
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2 scripts/inference.py configs/opensora-v1-3/inference/t2v.py \
  --num-frames 49 --resolution 360p --aspect-ratio 9:16 \
  --prompt-path ucf.txt

# webvid
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node 4 scripts/inference.py configs/opensora-v1-3/inference/t2v.py \
  --num-frames 49 --resolution 360p --aspect-ratio 9:16 \
  --prompt-path webvid.txt

```


## Evaluation

```bash
python metrics.py --text physvideobench.txt --video_dir samples/ --metrics FVD dJ L_div
python metrics.py --text ucf.txt --video_dir samples/ --metrics FVD frame_consistency
python metrics.py --text webvid.txt --video_dir samples/ --metrics FVD clip_sim
```