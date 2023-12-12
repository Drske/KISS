# KISS
Klustering Images for Subset Selection. Research on the challenge of selecting a representative training subset for convolutional neural classifiers, employing methods that leverage clustering and difficulty estimation for specialized sample selection. Explore various techniques, experiments, and tools designed to address this problem.

## About
In this project, our goal is to design and implement efficient methods for selecting a training subset from a larger image dataset. The chosen subsets should be relatively small while maintaining or slightly compromising classification accuracy.

To accomplish this, we leverage [DINOv2](https://arxiv.org/abs/2304.07193) as our advanced feature extractor, providing vectorized representations of the images. Subsequently, we employ the [FAISS](https://arxiv.org/abs/1702.08734) library to execute multiple similarity searches and clusterings, determining the most representative subset of elements.

The repository includes methods and experiments validating their performance, positioning the project as both a practical tool and a versatile research framework.

P.S. We are aware it's clustering, not *klustering*.

## Setup
To use our project, follow these simple steps:

### Install PyTorch
Visit the [PyTorch](https://pytorch.org) site to select your system configuration. We recommend using `conda` and the `Nightly` build:

```bash
conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly
```

### Clone the GitHub repository:
```bash
git clone https://github.com/Drske/KISS.git
```

### Build the `kiss` package
```bash
pip install .
```

For an editable installation, run:

```bash
pip install -e .
```

### Test
Finally, execute the `kiss hello` command to verify if the package has been installed correctly.

```bash
kiss hello
```

## Repository structure

### data
This directory serves as a placeholder for any dataset that should be downloaded while using the repository.

### experiments
All experiment configurations and results are stored here.

### src
Contains the source code of the `kiss` project.

### notebooks
Explore useful notebooks, including examples or prototypes.

### checkpoints
All pretrained model weights are stored here.


## License
Our package has been released under the MIT License. Refer to [LICENSE](LICENSE) for more details.