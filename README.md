# DeepCurer

**Pruning-based Backdoor Mitigation via Progressive Neuron Ranking using Adversarial Proxies**

DeepCurer is a defense against backdoor (trojan) attacks on deep neural networks. Given a
potentially poisoned model, it locates and removes the neurons responsible for the backdoor
behavior while preserving clean accuracy — without needing access to poisoned training data or
knowledge of the trigger.

The core idea is to substitute hard-to-obtain triggered inputs with **adversarial proxies**, then
rank neurons by an **ASR/CA impact ratio** (how much each neuron contributes to attack success
relative to its contribution to clean accuracy). Neurons are pruned **progressively** from the most
backdoor-associated down, so the model is cleaned with minimal loss of utility.

This repository contains sample code for the paper. It is configured to run out of the box on
CIFAR-10 and extends to Tiny-ImageNet and GTSRB.

---

## Method Overview

1. **Adversarial proxy generation** — Instead of relying on real triggered samples, DeepCurer
   crafts adversarial examples that emulate the effect a backdoor trigger has on the network's
   internal activations.
2. **Progressive neuron ranking** — Each neuron is scored by its impact on attack success rate
   (ASR) versus clean accuracy (CA). Neurons with a high ASR/CA impact ratio are the strongest
   candidates for carrying the backdoor.
3. **Pruning** — Neurons are pruned in ranked order, tracking ASR and CA at each step, until the
   backdoor is neutralized while clean accuracy is retained.

---

## Repository Structure

| File | Purpose |
|------|---------|
| `cifar10.py` | Downloads CIFAR-10 and arranges it into train/test folders. |
| `backdoor_triggers.py` | Trigger definitions for the supported attacks. |
| `poison_dataset_generator.py` | Builds poisoned datasets from clean data + triggers. |
| `backdoor_training.py` | Trains a backdoored model for a chosen attack (`--atk`). |
| `generating_adverserial_examples.py` | Generates the adversarial proxy set used for ranking. |
| `pruning.py` / `pruning_only.py` | Pruning routines. |
| `deepcurer.py` | Main defense: proxy generation, neuron ranking, and progressive pruning. |
| `resnet.py`, `vgg.py` | Model architectures. |
| `train_test.py` | Training / evaluation utilities. |
| `sig.pt`, `pattern25.png`, `trojnn.jpg` | Trigger assets used by specific attacks. |

---

## Requirements

- Python 3.8+
- PyTorch and torchvision
- NumPy, Matplotlib
- (plus the usual scientific-Python stack)

```bash
pip install torch torchvision numpy matplotlib
```

> Adjust the PyTorch install command to match your CUDA version — see
> https://pytorch.org/get-started/locally/.

---

## Usage

### 1. Prepare the dataset

```bash
python cifar10.py
```

This downloads CIFAR-10 and organizes it into train/test folders. To run on **Tiny-ImageNet** or
**GTSRB**, download the dataset into the working directory and organize it in the same folder
structure as CIFAR-10.

### 2. Train a backdoored model

```bash
python backdoor_training.py --atk badnet
```

Saves a checkpoint of the attacked model.

### 3. Clean the model with DeepCurer

```bash
python deepcurer.py --atk badnet
```

This prunes the backdoor neurons of the affected model and saves:

- the proxy adversarial dataset,
- the backdoor neuron ranking file,
- the pruning results as a text file, and
- the pruning progress as a plot.

---

## Supported Attacks

`badnet`, `wanet`, `blend`, `fiba`, `trojan`, `sig`, `cl`, `bppattack`, `filter`, `lira`

Pass any of these to the `--atk` argument of `backdoor_training.py` and `deepcurer.py`.

## Supported Datasets

CIFAR-10 (default), Tiny-ImageNet, GTSRB.

---

## Results

Across the evaluated backdoor attacks, DeepCurer drives the average attack success rate down to
roughly **3.7%** while retaining about **91.5%** clean accuracy. See the paper for the full
per-attack breakdown and comparisons against baseline defenses.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{deepcurer,
  title   = {DeepCurer: Pruning-based Backdoor Mitigation via Progressive Neuron Ranking using Adversarial Proxies},
  journal = {Neurocomputing},
  year    = {2026}
}
```

> Please fill in the authors, volume, pages, and DOI once available.

---

## Contact

Developed by the Silicon Systems Lab (SiSL), University of Rhode Island. For questions, please open
an issue on this repository.
