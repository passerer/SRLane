## Sketch and Refine: Towards Fast and Accurate Lane Detection
Currently still under development.
### Install
```bash
git clone https://github.com/passerer/SRLane.git
cd SRLane
conda create -n py38 python=3.8 -y # Create a new Python environment, optional.
conda activate py38 
pip install -r requirements.txt
pip install torch==1.13.0+cu117 # Install pytorch, modifying the CUDA version accordingly.
python setup.py develop
```
### DATASET
Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then modify `dataset_path` in [configs/datasets/culane.py](configs/datasets/culane.py) accordingly.
### Train
Here is an example
```bash
CUDA_VISIBLE_DEVICES=0 python tools/main.py configs/exp_srlane_culane.py
```
### Test
Here is an example

```bash
CUDA_VISIBLE_DEVICES=0 python tools/main.py configs/exp_srlane_culane.py --load_from checkpoint/baseline.pth --validate
```