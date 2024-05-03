# Video-LLaVA-sort-tool
custom script for Video-LLaVA that can sort videos based on their visual content

This script uses Video-LLaVA to sort videos by repeating the same question for each video, and then moves the videos into directories based on the response.

# WIP, noob dev here and I'm figuring out the install process

I recently made [CLIP-video-sorter](https://github.com/secretlycarl/CLIP-video-sorter) but was unhappy with the results. I found an alternative in [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) so I made this script.

# Requirements
Python >= 3.10

Pytorch == 2.0.1

[Anaconda](https://docs.anaconda.com/free/anaconda/install/)

CUDA Version >= 11.7 but less than 12.4 (Video-LLaVA requires bitsandbytes, which is not compatible with 12.4. [CUDA swap guide](https://github.com/bycloudai/SwapCudaVersionWindows))

Not sure about RAM or VRAM requirements. I have 12GB VRAM and 32GB RAM. I'd assume at least 8GB and 16GB are needed, respectively.

~18GB disk space (mostly the models needed to analyze the videos)

# Video-LLaVA Install
```
git clone https://github.com/PKU-YuanGroup/Video-LLaVA
cd Video-LLaVA
conda create -n videollava python=3.10 -y
conda activate videollava
pip install --upgrade pip
pip install -e .
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```
`pip install -e ".[train]"` and `pip install flash-attn --no-build-isolation` aren't needed just to run inference on a video in the CLI

Then copy my `sortscript.py` into the main Video-LLaVA directory

bitsandbytes fix?

load_in_8bit_fp32_cpu_offload=True fix?

# Usage
1. Activate the environment and run my script with `python sortscript.py`
The first time it runs, it will download ~17GB of models.

When trying to run on my system for the first time, after the models are downloaded, there is a warning about bitsandbytes and this error
``` 
Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit
the quantized model. If you want to dispatch the model on the CPU or the disk while keeping
these modules in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom
`device_map` to `from_pretrained`. Check
https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
for more details.
```
2. If you get this, just ignore it, close the command window, reopen, activate the environment, and run the script again. Trying to figure out how to deal with this before it happens but no luck yet.

3. It will ask you for a folder path. Copy and paste from the file explorer.

4. It will ask you for your sorting question. My questions are usually something like "What kind of X is this? 1 word answer". X could be landscape, dog, car, etc. A question like "What kind of X does Y have? 1 word answer" is good too.
I've found that letting the model pick how to describe things works better than giving it options to pick from, like "is this X, Y, or Z?" becuase it can interpret your options incorrectly. 

5. Next the script will loops through the folder of videos and provide its answer to the question for each video.

6. At the end, the videos are sorted into folders based on the model's answers.

7. It will ask for a new folder if you want to sort further.

In my initial tests my script worked but had 4 warnings. To fix 2 of them, I had to set temperature and top_p to 1 in `...Video-LLaVA\cache_dir\models--LanguageBind--Video-LLaVA-7B\snapshots\aecae02b7dee5c249e096dcb0ce546eb6f811806\generation_config.json`. This file doesn't show up in my file explorer and I only found it with the project folder loaded in [Cursor](https://cursor.sh/) after the script runs for the first time and errors out. I would share the updated file but I don't know how to integrate it from outside the environment. There are other warnings but they don't seem critical, though I wlecome any suggestions to fix them.
