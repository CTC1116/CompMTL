# CompMTL: Layer-Wise Competitive Multi-Task Learning

## Overview

**CompMTL (Layer-Wise Competitive Multi-Task Learning)** is a novel approach designed to address multi-task conflicts in shared networks by fine-tuning gradients at the layer level. By adjusting gradient updates based on the relative importance of tasks at each layer of the model, **CompMTL** aims to improve convergence and performance across multiple tasks. This method outperforms conventional multi-task learning approaches by mitigating conflicts and facilitating task-specific progress.

### Key Features:
- **Layer-wise Gradient Balancing:** Mitigates conflicts by adjusting task gradients at specific layers.
- **Improved Task Convergence:** Ensures that each task receives appropriate gradient updates based on layer-wise importance, leading to more efficient training.
- **Curriculum Learning Support:** Reduces training time by gradually enlarging the competitive space during training.
- **Integration with Existing Methods:** **CompMTL** can be combined with existing multi-task optimization strategies for enhanced performance.

## Paper

For a comprehensive explanation of the **CompMTL** method, including experimental results on multiple datasets, refer to the paper:  
[**CompMTL: Layer-Wise Competitive Multi-Task Learning**](https://ieeexplore.ieee.org/abstract/document/10888316)
*Published in ICASSP 2025.*

## Installation

### Prerequisites:
- **Python 3.6+**
- **PyTorch** (CUDA support is recommended for training with GPUs)
- Other dependencies are listed in the `requirements.txt` file.

### Setup Instructions:

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/CompMTL.git
    cd CompMTL
    ```

2. Download the required datasets (e.g., **Cityscapes**, **NYU-V2**) as per the instructions provided in the paper.

## Training

The **CompMTL** implementation uses a **SegFormer** model with a **MiT-B0** backbone. To train the model on the **Cityscapes** dataset with multi-task learning, use the following command:

```bash
python3 train_baseline_MTL_progress_KD.py --model segformer_multi --backbone MiT_B0 --task multi --flag com_mtl --dataset city_256 --data '/path/to/dataset/cityscapes256_512/' --batch-size 8 --val-batch 4 --max-iterations 100000 --lr 0.0001 --weight-decay 0.000001 --kd-weight 1 1 1 --sigma 10.0 --temp 1.0 --device cuda:2 --pretrained '../mit_b0.pth'
