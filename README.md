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
```

## Command-Line Arguments:
| Argument           | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `--model`          | **SegFormer** architecture for multi-task learning (or ResNet).          |
| `--backbone`       | **MiT-B0** backbone model.                                   |
| `--task`           | Specifies the multi-task setup.                              |
| `--flag`           | Activates **CompMTL** optimization strategy (or other works: GradNorm, PCGrad, AMTL, CAGrad, and so on).                 |
| `--dataset`        | Choose the dataset (e.g., **city\_256** for **Cityscapes**). |
| `--data`           | Path to the dataset.                                         |
| `--batch-size`     | Training batch size (default: 8).                            |
| `--val-batch`      | Validation batch size (default: 4).                          |
| `--max-iterations` | Maximum number of iterations (default: 100,000).             |
| `--lr`             | Learning rate (default: 0.0001).                             |
| `--weight-decay`   | Weight decay for regularization.                             |
| `--kd-weight`      | Knowledge distillation weights for multi-task distillation.  |
| `--sigma`          | Regularization coefficient (default: 10.0).                  |
| `--temp`           | Knowledge distillation temperature (default: 1.0).           |
| `--device`         | Specify the CUDA device (e.g., **cuda:2**).                  |
| `--pretrained`     | Path to pre-trained weights for the **MiT-B0** model.        |

## Citation
If you use CompMTL in your research, please cite the following paper:
```
@inproceedings{cheng2025compmtl,
  title={CompMTL: Layer-Wise Competitive Multi-Task Learning},
  author={Cheng, Tiancong and Zhang, Ying and Shah, Rajiv Ratn and Zimmermann, Roger and Yu, Zhiwen and Guo, Bin},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
