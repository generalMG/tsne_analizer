# 2D and 3D t-SNE analysis using Scikit-Learn and Torch ImageNet ResNet101

This repository contains a script to perform t-SNE analysis on image datasets using features extracted from a pre-trained ResNet101 model. The script processes images, extracts features, and visualizes the results in both 2D and 3D plots.

## Usage

```bash
Usage: python3 tsne_analysis.py <n_iter> <n_components> <image_dir> <perplexity>
```

### Parameters
- <n_iter>: Number of iterations for the t-SNE algorithm.
- <n_components>: Number of components for the t-SNE (2 or 3).
- <image_dir>: Directory containing the images, structured in subdirectories by class.
- <perplexity> (optional): Perplexity value for the t-SNE algorithm.

## Installation

Clone the repo.

```bash
git clone https://github.com/generalMG/tsne_analyzer.git
cd tsne_analyzer
```
## Running the Script

1. Prepare your dataset:
   - Organize your images into subdirectories named by class within a main directory or change the code for your own dataset structure.
2. Run the script:

```bash
python tsne_analysis.py 1000 2 /path/to/your/images 30
```

## Figures

### Example 1 - 2D
![tsne_P5_2D](https://github.com/user-attachments/assets/4686b1ae-de04-43aa-8709-47c2b21a5500)

### Example 2 - 3D
![tsne_P5_3D](https://github.com/user-attachments/assets/8c7b396f-df87-4d80-8609-60726ab829f6)
