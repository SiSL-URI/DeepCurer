# DeepCurer (Neurocomputing 2026)

**Pruning-based Backdoor Mitigation via Progressive Neuron Ranking using Adversarial Proxies**

Backdoor attack, where an adversary poisons the model by misclassifying inputs into an attacker-defined label, shows its uprising threats on deep learning models. Detecting and pruning backdoor neurons has emerged as an effective defense; however, precise mitigation of backdoor neurons from benign ones remains a challenge. Most existing works focus on exploiting empirical characteristics of backdoor neurons, such as their sensitivity to adversarial perturbations or asymmetric learning behavior, while overlooking their theoretical definition. In this paper, we propose a novel backdoor neuron ranking and pruning framework named DeepCurer, where we quantify each neuron’s impact on both clean and backdoor tasks and rank-and-prune them based on the inherent definition of backdoor neurons. To enable effective use of our method in the absence of poisoned samples, we demonstrate that targeted adversarial examples can serve as proxies. We also introduce a lightweight technique to detect target labels using adversarial proxies. A progressive neuron rank-and-prune algorithm is further developed to systematically remove backdoor neurons and sanitize underlying deep learning models. Experimental results show our proposed method can effectively mitigate backdoor neurons against ten state-of-the-art backdoor attacks at ultra-low poisoning ratios (
=1%) while preserving the model’s clean performance. DeepCurer also upholds its superior performance by comparing against other state-of-the-art backdoor defense methods. 

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
@article{miah2026deepcurer,
  title={DeepCurer: Pruning-based backdoor mitigation via progressive neuron ranking using adversarial proxies},
  author={Miah, Abdullah Arafat and Bi, Yu},
  journal={Neurocomputing},
  pages={134409},
  year={2026},
  publisher={Elsevier}
}
```


---

## Contact

Please contact at {abdullaharafat.miah , yu_bi}@uri.edu for any issues.
