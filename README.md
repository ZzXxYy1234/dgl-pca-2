# DGL-PCA

PyTorch implementation of DGL-PCA for multimodal conversational emotion recognition.

```text
dgl_pca/      Model, data loading, training, and evaluation code
configs/      CMU-MOSEI and IEMOCAP experiment configurations
data/         Local prepared datasets
checkpoints/  Local model checkpoints
train.py      Training entry point
eval.py       Evaluation entry point
tests/        Unit tests
```

## Setup

```bash
conda env create -f environment.yml
conda activate dgl-pca
pip install -e .
```

## Train

```bash
python train.py --config configs/mosei_sentiment_2.yaml --seed 24 --device cuda:0
```

Validate a configuration without loading data:

```bash
python train.py --config configs/mosei_sentiment_2.yaml --seed 24 --dry-run
```

## Evaluate

```bash
python eval.py \
  --config configs/mosei_sentiment_2.yaml \
  --checkpoint checkpoints/mosei/mosei_sentiment_2_seed24.pt \
  --seed 24 \
  --device cuda:0
```

Run a quick check:

```bash
python eval.py --smoke-test
python -m unittest discover -s tests
```

## IEMOCAP

The repository includes IEMOCAP training and evaluation configurations only. It
does not include IEMOCAP data or checkpoints. The configurations expect
pre-extracted 100-dimensional acoustic, 768-dimensional textual, and
512-dimensional visual features. They use Session 5 for testing; the 4-way
setting retains Happy, Sad, Neutral, and Angry only. Prepare the required local
dataset file before running one of these commands:

```bash
python train.py --config configs/iemocap_4.yaml --seed 24 --device cuda:0
python eval.py --config configs/iemocap_6.yaml --checkpoint checkpoints/iemocap/iemocap_6_seed24.pt --device cuda:0
```
