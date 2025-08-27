# [CVPR'25 Highlight] Graph-Generating State Space Models (GG-SSMs)
**Official PyTorch Implementation of our [CVPR 2025 Highlight paper](https://arxiv.org/abs/2412.12423).**  
**Authors:** Nikola Zubić, Davide Scaramuzza

Please, check the poster [here](https://github.com/uzh-rpg/gg_ssms/blob/master/CVPR25_Zubic_poster.pdf).

---

<!-- Insert your image here -->
<p align="center">
  <img src="https://github.com/uzh-rpg/gg_ssms/blob/master/gg_ssm_core.png" alt="GG-SSM Overview" width="500px"/>
</p>
<p align="center">
  <em>Figure: Overview of the GG-SSM pipeline applied to various tasks, such as event-based vision tasks, 
  time series forecasting, image classification, and optical flow estimation.</em>
</p>

## Abstract
State Space Models (SSMs) are powerful tools for modeling sequential data in computer vision and time series analysis domains. However, traditional SSMs are limited by fixed, one-dimensional sequential processing, which restricts their ability to model non-local interactions in high-dimensional data. While methods like Mamba and VMamba introduce selective and flexible scanning strategies, they rely on predetermined paths, which fails to efficiently capture complex dependencies.

We introduce **Graph-Generating State Space Models (GG-SSMs)**, a novel framework that overcomes these limitations by dynamically constructing graphs based on feature relationships. Using Chazelle's Minimum Spanning Tree algorithm, GG-SSMs adapt to the inherent data structure, enabling robust feature propagation across dynamically generated graphs and efficiently modeling complex dependencies.

We validate GG-SSMs on 11 diverse datasets, including event-based eye-tracking, ImageNet classification, optical flow estimation, and six time series datasets. GG-SSMs achieve state-of-the-art performance across all tasks, surpassing existing methods by significant margins. Specifically, GG-SSM attains a top-1 accuracy of **84.9%** on ImageNet, outperforming prior SSMs by **1%**, reducing the KITTI-15 error rate to **2.77%**, and improving eye-tracking detection rates by up to **0.33%** with fewer parameters.
These results demonstrate that dynamic scanning based on feature relationships significantly improves SSMs' representational power and efficiency, offering a versatile tool for various applications in computer vision and beyond.

## Citation
If you find this work helpful, please cite our paper:
```bibtex
@inproceedings{Zubic_2025_CVPR,
  title     = {Graph-Generating State Space Models (GG-SSMs)},
  author    = {Zubic, Nikola and Scaramuzza, Davide},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```

## Standalone Installation

Below are the commands to set up a conda environment and install all necessary dependencies, including custom libraries for graph-based state scanning:
```bash
# 1. Create and activate conda environment
conda create -y -n gg_ssms python=3.11
conda activate gg_ssms

# 2. Install PyTorch and CUDA
conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -y nvidia::cuda-toolkit

# 3. Install custom dependencies (TreeScan and TreeScanLan)
cd core/convolutional_graph_ssm/third-party/TreeScan/
pip install -v -e .
cd $(git rev-parse --show-toplevel)

cd core/graph_ssm/third-party/TreeScanLan/
pip install -v -e .
```

## Additional Dependencies
Depending on which tasks or modules you want to run, you may need extra Python packages beyond the core requirements listed above. Below is a breakdown of recommended installations for each sub-project:
1. **INI-30 dataset event-based eye tracking (eye_tracking_ini_30)**
   ```bash
   cd eye_tracking_ini_30
   pip install dv-processing sinabs tonic thop samna fire
   ```
2. **LPW dataset event-based eye tracking (eye_tracking_lpw)**
   ```bash
   cd eye_tracking_lpw
   pip install matplotlib opencv-python tqdm tables easydict wandb timm einops
   ```
3. **MambaTS (Time Series)**
   - Check the [requirements.txt](MambaTS/requirements.txt) inside the `MambaTS` folder:
   ```bash
   cd MambaTS
   pip install -r requirements.txt
   ```
## General Usage with Pretrained Models

### Convolutional Graph SSM for Image-Based Tasks
We provide a **Convolutional Graph-Generating SSM** for image-based feature extraction and classification in:
```
core/convolutional_graph_ssm/classification/models/graph_ssm.py
```
- **Choosing Model Size**: On **line 545**, you can set `config_path` to one of `base`, `small`, or `tiny` to pick the desired model variant.  
- **Pretrained Weights**: Place the corresponding pretrained weight files (e.g., `gg_ssm_base.pth`, `gg_ssm_small.pth`, `gg_ssm_tiny.pth`) inside:
  ```
  core/convolutional_graph_ssm/classification/weights/
  ```
  These weights can be downloaded from the [Releases](https://github.com/uzh-rpg/gg_ssms/releases) page.

#### Example Usage
To run a forward pass on an image:
```bash
python core/convolutional_graph_ssm/classification/models/graph_ssm.py
```
- By default, this script will load the **base** model from `config_path='base'`.  

### Temporal Graph SSM
A purely **temporal** Graph-Generating SSM (for sequential or time-series data) is available in:
```
core/graph_ssm/main.py
```
- This module focuses on modeling temporal dependencies using dynamically constructed graphs.

### Spatio-Temporal Usage
You can **combine** the **Convolutional Graph SSM** (for spatial modeling) and the **Temporal Graph SSM** (for sequential/temporal modeling) to create a unified spatio-temporal pipeline. Our event-based eye tracking tasks (see [Ini-30 Eye Tracking](#ini-30-eye-tracking) or [LPW Dataset Eye Tracking](#lpw-dataset-eye-tracking)) demonstrate exactly how these two components are integrated for end-to-end training.

## Time Series Tasks
We incorporate **Graph-Generating SSMs** into the **MambaTS** codebase by replacing the default encoder in `MambaTS/models/MambaTS.py` with our `TemporalGraphSSM`. This allows graph-based temporal modeling for long-horizon forecasting.

### How to Run
1. **Scripts Location**  
   All relevant scripts can be found [here](https://github.com/uzh-rpg/gg_ssms/tree/master/MambaTS/scripts).
   
3. **Adjusting Paths & Parameters**  
   In each script (e.g., `run.py`), you can modify:
   - **`CUDA_VISIBLE_DEVICES`**: Set to your GPU index (e.g., `export CUDA_VISIBLE_DEVICES=3`).
   - **`root_path` / `data_path`**: Point these to the folder containing your time-series dataset.  
   - **`model_id` / `model_name`**: Namespacing for checkpoints and logging.
   - **`seq_len`, `pred_len`**: Sequence length and prediction horizon you want to experiment with.
   - **Hyperparameters**: Adjust `e_layers`, `d_layers`, `batch_size`, `learning_rate`, etc.

4. **Datasets Download**
All datasets can be downloaded from here.
   
To run, do `cd MambaTS/` from the root and then `bash ./scripts/MambaTS_ETTh2.sh` to run ETTh2 dataset training. All the other scripts for any of the 6 time series datasets are available. All the logs and outputs will be generated inside the MambaTS folder.

## Ini-30 Eye Tracking
Our implementation for Ini-30 event-based eye tracking can be found in the `retina` folder:
- **`/training/models/baseline_3et.py`**:  
  Contains the code where our **GG-SSM** architecture is integrated for eye tracking with a `spatial_backbone=ConvGraphSSM` and `temporal_ssm=TemporalGraphSSM`.
- From the root you can run `CUDA_VISIBLE_DEVICES=i python retina/scripts/train.py --run_name=graph_ssm --device=i`, where i is the GPU ID.
The script will automatically log and create a project in **Weights & Biases (wandb)**, named `eye_tracking_ini_30`.

### Tonic & NumPy Version Conflicts
When installing **Tonic** (needed for event-based data processing), you may encounter a pip dependency error like:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
...
python-tsp 0.5.0 requires numpy<3.0.0,>=2.0.0, but you have numpy 1.26.4 which is incompatible.
```
This means **Tonic** and **python-tsp** (used for certain Time Series tasks) have **conflicting NumPy requirements**. If you plan to run **Time Series tasks** in the same environment, you can:
1. **Uninstall Tonic** once finished with eye tracking,  
2. Downgrade or reinstall NumPy, and  
3. Reinstall `python-tsp` for time series.

Alternatively, keep separate environments for each task to avoid conflicts.

## LPW Dataset Eye Tracking
Our integration for the LPW dataset eye tracking is located in the `eye_tracking_lpw` folder.

1. **Data Preparation**  
   Follow the instructions provided by [cb-convlstm-eyetracking](https://github.com/qinche106/cb-convlstm-eyetracking/) to download and prepare the LPW dataset.

2. **Path Configuration**  
   In the `eye_tracking_lpw/graph_ssm_train.py` file, set:
   ```python
   DATA_DIR_ROOT = "/path/to/your/LPW/dataset"
   ```
   so that it points to the root directory containing the LPW dataset.

3. **Run Training**  
   From the project root directory, simply execute:
   ```bash
   python eye_tracking_lpw/graph_ssm_train.py
   ```
   This will start the training process for LPW eye tracking with the **Graph-Generating SSM** architecture.

## Code Acknowledgments
This project has used code from the following projects:
- [MambaTS](https://github.com/XiudingCai/MambaTS-pytorch) - Improved Selective State Space Models for Long-term Time Series Forecasting
- [Retina](https://github.com/pbonazzi/retina) - Low-Power Eye Tracking with Event Camera and Spiking Hardware
- [3ET](https://github.com/qinche106/cb-convlstm-eyetracking) - Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network
- [MemFlow](https://github.com/DQiaole/MemFlow) - Optical Flow Estimation and Prediction with Memory
- [GrootVL](https://github.com/EasonXiao-888/GrootVL) - Tree Topology is All You Need in State Space Model
