Sample code for the 'DeepCurer: Pruning-based Backdoor Mitigation via Progressive Neuron Ranking using Adversarial Proxies'

Instructions for running the code:

01. Run 'cifar10.py' to download cifar 10 dataset and arrange it train and test folders. This sample code will run for cifar 10. If you want to run for 'tiny-imagenet' and 'gtsrb' then please download the dataset in the working directory and organise it as 'cifar 10'. 

02. Run 'backdoor_training.py' to get the checkpoint of an attack.

    Example: run backdoor_training.py --atk 'badnet'

    Attack choices: 'badnet', 'wanet', 'blend', 'fiba', 'trojan', 'sig', 'cl', 'bppattack', 'filter', 'lira'.

03. Run 'deepcurer.py' to prune backdoor neurons of an affected model. It will save the proxy adverseraial dataset, backdoor neuron ranking file, the pruning results as a text file, and pruning progress as a    plot.

   Example: run deepcurer.py --atk 'badnet'
   Attack choices: 'badnet', 'wanet', 'blend', 'fiba', 'trojan', 'sig', 'cl', 'bppattack', 'filter', 'lira'.


