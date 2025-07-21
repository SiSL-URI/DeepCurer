import torch
import torch.nn as nn
from tqdm import tqdm
import train_test

def evaluate_bn_neurons_full_testset(net, testloader, poisoned_testloader, cag, asrg, epsilon=1e-8):
    """
    Fast BN neuron evaluation using full test set, with forward hooks instead of modifying BN weights.
    Adds a small epsilon to denominator to avoid infinity values.
    """
    device = next(net.parameters()).device
    net.eval()

    bn_neuron_effects = []

    # Store original CA and ASR with full test set
    CA_ref = train_test.test(net, testloader)
    ASR_ref = train_test.test(net, poisoned_testloader)

    # Iterate over BN layers
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            print(f"Evaluating BN layer: {name}")
            num_neurons = module.weight.size(0)

            for neuron_idx in tqdm(range(num_neurons), desc=f"Processing neurons in {name}"):
                # Define a forward hook to zero-out neuron_idx
                def hook_fn(module, input, output):
                    output[:, neuron_idx, :, :] = 0
                    return output

                handle = module.register_forward_hook(hook_fn)

                # Evaluate CA and ASR with neuron masked
                CA_masked = train_test.test(net, testloader)
                ASR_masked = train_test.test(net, poisoned_testloader)

                handle.remove()  # Important: remove hook after inference

                # Calculate diffs
                CA_diff = abs(CA_ref - CA_masked)
                ASR_diff = abs(ASR_ref - ASR_masked)

                # Add epsilon to denominator to avoid infinity
                ratio = ASR_diff / (CA_diff + epsilon)

                # Save effects
                bn_neuron_effects.append({
                    'layer': name,
                    'neuron': neuron_idx,
                    'CA diff': CA_diff,
                    'ASR diff': ASR_diff,
                    'ASR/CA': ratio
                })

    # Sort based on ASR/CA ratio descending
    ranked_neurons = sorted(bn_neuron_effects, key=lambda x: x['ASR/CA'], reverse=True)

    return ranked_neurons

