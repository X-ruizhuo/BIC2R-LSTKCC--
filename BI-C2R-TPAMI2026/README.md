# Bi-C2R with Short-Term Old Model C-STKR

This repository is based on **Bi-C2R: Bidirectional Continual Compatible Representation for Re-indexing Free Lifelong Person Re-identification**.

The current version adds a lightweight LSTKC++-style enhancement to the new-model training branch: a short-term old model is preserved before long-term model fusion, and the long-term/short-term old models jointly build a C-STKR affinity target to supervise the current model with KL divergence.

## Baseline

The original Bi-C2R framework contains:

- A ResNet-50 ReID backbone with BNNeck and GeM pooling.
- Bidirectional Compatible Transfer Networks (BiCT-Net) for old-to-new and new-to-old feature transfer.
- Bidirectional Compatible Distillation (BiCD) for feature and relation compatibility.
- Bidirectional Anti-forgetting Distillation (BiAD) for discriminative knowledge preservation.
- Dynamic Feature Fusion (DFF) for adaptive model and historical gallery feature fusion.

The default training orders are:

- Setting 1: `market1501 -> cuhk_sysu -> dukemtmc -> msmt17 -> cuhk03`
- Setting 2: `dukemtmc -> msmt17 -> market1501 -> cuhk_sysu -> cuhk03`

## Added Design

At stage `t`, the code now distinguishes two old models:

- `old_model`: the long-term old model, i.e. the fused model from previous stages. It keeps the original Bi-C2R role and still drives BiCT, BiCD, BiAD, DFF, and gallery feature updating.
- `short_old_model`: the short-term old model, i.e. the previous stage model before long-term fusion. It is used only to generate the C-STKR KL supervision target.

The current model branch is supervised as follows:

1. Extract current features with the current model.
2. Extract old features with the long-term old model.
3. Extract short-term old features with the short-term old model.
4. Build long-term and short-term affinity matrices on the current batch.
5. Apply STKR-style threshold rectification to both old affinity matrices.
6. Fuse the two rectified matrices with C-STKR:
   - if both old models are correct or both are wrong for a pair, average the two rectified scores;
   - if only one old model is correct, keep the correct model's rectified score.
7. Use KL divergence from the current affinity matrix to the C-STKR target as the anti-forgetting supervision.

If `short_old_model` is unavailable, training falls back to the original single-old-model KL target.

## Key Code Paths

- `continual_train.py`
  - Stores `short_old_model_candidate = copy.deepcopy(model)` after each stage training and before long-term fusion.
  - Passes `short_old_model` into `train_dataset`.
- `reid/trainer.py`
  - Adds `short_old_model` to `Trainer.train`.
  - Adds `build_stkr_target`, `build_cstkr_target`, and `cal_CSTKR_KL`.
  - Keeps the original `cal_KL` fallback.
- `tests/test_short_old_model_cstkr_static.py`
  - Checks that the short-term old model and C-STKR interfaces remain wired into the training flow.

## Installation

```shell
conda create -n IRL python=3.7
conda activate IRL
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirement.txt
```

## Prepare Datasets

Download the person re-identification datasets Market-1501, MSMT17, CUHK03, and the other supported datasets following the original Bi-C2R data preparation. Place them under the configured `--data-dir` root.

The default scripts expect the dataset root to contain folders such as:

```text
PRID/
  CUHK03/
  CUHK-SYSU/
  DukeMTMC-reID/
  MSMT17_V2/
  Market-1501/
```

## Training and Evaluation

```shell
bash run1.sh
bash run2.sh
```

The new C-STKR branch is enabled by default because it only activates when a `short_old_model` exists. The first stage has no old model. From the second stage onward, the previous stage's pre-fusion model is used as the short-term old model.

## Logged Losses

Training logs and TensorBoard scalars record the original Bi-C2R loss terms plus the added C-STKR supervision:

- `Loss_ce`: identity classification loss.
- `Loss_tr`: triplet loss.
- `Loss_ca`: BiCD-compatible feature alignment loss.
- `Loss_cr`: BiCD relation distillation loss.
- `Loss_ad`: BiAD discriminative distillation loss.
- `Loss_dc`: BiAD directional consistency loss.
- `Loss_cstkr`: KL divergence between the current affinity matrix and the C-STKR rectified long/short-term old-model affinity target.

## Validation

Static interface validation can be run without the full CUDA training environment:

```shell
python -m pytest tests/test_short_old_model_cstkr_static.py -q
python -m py_compile continual_train.py reid/trainer.py
```

Full numerical validation requires the original project environment with PyTorch and datasets:

```shell
bash run1.sh
bash run2.sh
```

Recommended ablations:

- Original Bi-C2R.
- Bi-C2R + short-term old model C-STKR.
- C-STKR loss weight sweep through `--AF_weight`.
- Compare L-ReID, RFL-ReID, and Average Forgetting after each stage.
