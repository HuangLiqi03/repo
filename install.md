# requirement.txt
保持gym torch maniskill2等库版本一致
------------------------ -------------
absl-py                  2.3.1
anyio                    4.11.0
appdirs                  1.4.4
arm_pytorch_utilities    0.4.3
asttokens                3.0.0
beautifulsoup4           4.14.2
black                    22.10.0
cached-property          2.0.1
cachetools               4.2.4
certifi                  2025.10.5
cffi                     2.0.0
charset-normalizer       3.4.4
click                    8.1.8
cloudpickle              1.3.0
cycler                   0.12.1
Cython                   3.1.6
dacite                   1.9.2
decorator                4.4.2
dm-control               0.0.403778684
dm-env                   1.6
dm-tree                  0.1.8
docker-pycreds           0.4.0
docstring_parser         0.17.0
eval_type_backport       0.2.2
exceptiongroup           1.3.0
executing                2.2.1
Farama-Notifications     0.0.4
fast_kinematics          0.2.2
fasteners                0.20
filelock                 3.19.1
fsspec                   2025.9.0
future                   1.0.0
gdown                    5.2.0
gitdb                    4.0.12
GitPython                3.1.45
glfw                     2.5.4
google-auth              1.35.0
google-auth-oauthlib     0.4.6
grpcio                   1.76.0
gym                      0.21.0
gym-notices              0.1.0
gymnasium                1.1.1
h11                      0.16.0
h5py                     3.14.0
hf-xet                   1.2.0
httpcore                 1.0.9
httpx                    0.28.1
huggingface-hub          1.0.1
idna                     3.11
imageio                  2.21.2
imageio-ffmpeg           0.6.0
importlib_metadata       8.7.0
ipython                  8.18.1
jedi                     0.19.2
Jinja2                   3.1.6
joblib                   1.5.2
kiwisolver               1.4.7
labmaze                  1.0.6
lxml                     6.0.2
Mako                     1.3.10
mani-skill2              0.4.1
Markdown                 3.9
markdown-it-py           3.0.0
MarkupSafe               3.0.3
matplotlib               3.4.2
matplotlib-inline        0.2.1
mdurl                    0.1.2
metaworld                0.0.1.dev0
moviepy                  1.0.3
mplib                    0.1.1
mpmath                   1.3.0
mujoco-py                2.1.2.14
mypy_extensions          1.1.0
networkx                 3.2.1
numpy                    1.23.2
nvidia-cublas-cu11       11.11.3.6
nvidia-cuda-cupti-cu11   11.8.87
nvidia-cuda-nvrtc-cu11   11.8.89
nvidia-cuda-runtime-cu11 11.8.89
nvidia-cudnn-cu11        9.1.0.70
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.3.0.86
nvidia-cusolver-cu11     11.4.1.48
nvidia-cusparse-cu11     11.7.5.86
nvidia-ml-py             13.580.82
nvidia-nccl-cu11         2.21.5
nvidia-nvtx-cu11         11.8.86
oauthlib                 3.3.1
opencv-python            4.11.0.86
packaging                25.0
parso                    0.8.5
patchelf                 0.17.2.1
pathspec                 0.12.1
pathtools                0.1.2
pexpect                  4.9.0
pillow                   11.3.0
pip                      23.3.2
platformdirs             4.4.0
proglog                  0.1.12
prompt_toolkit           3.0.52
protobuf                 4.25.8
psutil                   7.1.2
ptyprocess               0.7.0
pure_eval                0.2.3
pyasn1                   0.6.1
pyasn1_modules           0.4.2
pybullet                 3.2.5
pycparser                2.23
pygame                   2.6.1
Pygments                 2.19.2
pynvml                   13.0.1
PyOpenGL                 3.1.10
pyparsing                3.2.5
pyperclip                1.11.0
PySocks                  1.7.1
python-dateutil          2.9.0.post0
pytorch-kinematics       0.7.5
pytorch-seed             0.2.0
PyYAML                   6.0.3
requests                 2.32.5
requests-oauthlib        2.0.0
rich                     14.2.0
rsa                      4.9.1
rtree                    1.4.1
sapien                   2.2.1
scikit-video             1.1.11
scipy                    1.13.1
sentry-sdk               2.42.1
setproctitle             1.3.7
setuptools               64.0.3
shellingham              1.5.4
shtab                    1.7.2
six                      1.17.0
smmap                    5.0.2
sniffio                  1.3.1
soupsieve                2.8
stack-data               0.6.3
sympy                    1.14.0
tabulate                 0.9.0
tensorboard              2.20.0
tensorboard-data-server  0.7.2
tensorboard-plugin-wit   1.8.1
tomli                    2.3.0
toppra                   0.6.3
torch                    2.7.1+cu118
torchaudio               2.7.1+cu118
torchvision              0.22.1+cu118
tqdm                     4.64.0
traitlets                5.14.3
transforms3d             0.4.2
trimesh                  4.9.0
triton                   3.3.1
typeguard                4.4.4
typer-slim               0.20.0
typing_extensions        4.15.0
tyro                     0.9.35
urllib3                  2.5.0
wandb                    0.15.3
wcwidth                  0.2.14
Werkzeug                 3.1.3
wheel                    0.45.1
zipp                     3.23.0


# 修改后的experiment和environment文件（haowen）
备份在repocopy下


# render报错
It is usually due to the broken vulkan driver. Can you try to run vulkaninfo to show whether the vulkan driver is working?vulkaninfo can be installed through sudo apt-get install vulkan-utils
If vulkaninfo fails to show the information about Vulkan, you need to create /usr/share/vulkan/icd.d/nvidia icd.json and set theenvironment variable VK ICD FILENAMES=/usr/share/vulkan/icd.d/nvidia icd.json .
The content of nvidia icd.json :
```json
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155"
    }
}
```


# train
{PushCubeMatterport, LiftCubeMatterport, TurnFaucetMatterport}
python experiments/train_repo.py --algo repo --env_id maniskill-TurnFaucetMatterport --expr_name benchmark  --seed 0 --gpu_id 0

# adapt
python experiments/adapt_repo.py --algo repo_calibrate --env_id maniskill-LiftCubeMatterport --expr_name adaptaion --source_dir logdir/repo/maniskill-LiftCubeMatterport/benchmark/0 --seed 0 --alignment_mode distribution

# mani3
装环境要装最新的torch pip3的那个