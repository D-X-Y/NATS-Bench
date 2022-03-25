# [NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size](https://arxiv.org/abs/2009.00437)

Xuanyi Dong, Lu Liu, Katarzyna Musial, Bogdan Gabrys

in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2021

**Abstract**: Neural architecture search (NAS) has attracted a lot of attention and has been illustrated to bring tangible benefits in a large number of applications in the past few years. Network topology and network size have been regarded as two of the most important aspects for the performance of deep learning models and the community has spawned lots of searching algorithms for both of those aspects of the neural architectures. However, the performance gain from these searching algorithms is achieved under different search spaces and training setups. This makes the overall performance of the algorithms incomparable and the improvement from a sub-module of the searching model unclear.
In this paper, we propose NATS-Bench, a unified benchmark on searching for both topology and size, for (almost) any up-to-date NAS algorithm.
NATS-Bench includes the search space of 15,625 neural cell candidates for architecture topology and 32,768 for architecture size on three datasets.
We analyze the validity of our benchmark in terms of various criteria and performance comparison of all candidates in the search space.
We also show the versatility of NATS-Bench by benchmarking 13 recent state-of-the-art NAS algorithms on it. All logs and diagnostic information trained using the same setup for each candidate are provided.
This facilitates a much larger community of researchers to focus on developing better NAS algorithms in a more comparable and computationally effective environment.

**You can use `pip install nats_bench` to install the library of NATS-Bench
or install from source by `pip install .`.**

If you are seeking how to re-create NATS-Bench from scratch or reproduce benchmarked results, please see use [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects) and see these [instructions](https://github.com/D-X-Y/NATS-Bench#how-to-re-create-nats-bench-from-scratch).

If you have questions, please ask at [here](https://github.com/D-X-Y/NATS-Bench/issues) or [email me](mailto:dongxuanyi888@gmail.com) :)

This figure is the main difference between `NATS-Bench`, `NAS-Bench-101`, and `NAS-Bench-201`. The `topology search space` (`$\mathcal{S}_t$`) in `NATS-Bench` is the same as `NAS-Bench-201`, while we upgrade with results of more runs for the architecture candidates, and the benchmarked NAS algorithms have better hyperparameters.
<p align="center">
<img src="https://xuanyidong.com/resources/images/NATS-compare.png" width="700"/>
</p>


## Preparation and Download

**Step-1: download raw vision datasets.** (you can skip this one if you do not use weight-sharing NAS or re-create NATS-Bench).

In NATS-Bench, we (create and) use three image datasets -- CIFAR-10, CIFAR-100, and ImageNet16-120.
For more details, please see Sec-3.2 in [the NATS-Bench paper](https://arxiv.org/pdf/2009.00437.pdf). To download these three datasets, please find them at [Google Drive](https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4?usp=sharing).
To create the `ImageNet16-120` PyTorch dataset, please call [AutoDL-Projects/lib/datasets/ImageNet16](https://github.com/D-X-Y/AutoDL-Projects/blob/main/lib/datasets/get_dataset_with_transform.py#L168), by using:
```
train_data = ImageNet16(root, True , train_transform, 120)
test_data  = ImageNet16(root, False, test_transform , 120)
```

**Step-2: download benchmark files of NATS-Bench.**

The **latest** benchmark file of NATS-Bench can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1zjB6wMANiKwB2A1yil2hQ8H_qyeSe2yt?usp=sharing).
After download `NATS-[tss/sss]-[version]-[md5sum]-simple.tar`, please uncompress it by using `tar xvf [file_name]`.
We highly recommend to put the downloaded benchmark file (`NATS-sss-v1_0-50262.pickle.pbz2` / `NATS-tss-v1_0-3ffb9.pickle.pbz2`) or uncompressed archive (`NATS-sss-v1_0-50262-simple` / `NATS-tss-v1_0-3ffb9-simple`) into `$TORCH_HOME`.
In this way, our api will automatically find the path for these benchmark files, which are convenient for the users. Otherwise, you need to indicate the file when creating the benchmark instance manually.

The history of benchmark files is as follows, `tss` indicates the topology search space and `sss` indicates the size search space.
The benchmark file is used when creating the NATS-Bench instance with `fast_mode=False`.
The archive is used when `fast_mode=True`, where `archive` is a directory containing 15,625 files for tss or contains 32,768 files for sss.
Each file contains all the information for a specific architecture candidate.
The `full archive` is similar to `archive`, while each file in `full archive` contains **the trained weights**.
Since the full archive is too large, we use `split -b 30G file_name file_name` to split it into multiple 30G chunks.
To merge the chunks into the original full archive, you can use `cat file_name* > file_name`.

|   Date     |  benchmark file (tss) | archive (tss) | full archive (tss) |       benchmark file (sss)      |       archive (sss)        | full archive (sss) |
|:-----------|:---------------------:|:-------------:|:------------------:|:-------------------------------:|:--------------------------:|:------------------:|
| 2020.08.31 | [NATS-tss-v1_0-3ffb9.pickle.pbz2](https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing) | [NATS-tss-v1_0-3ffb9-simple.tar](https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view?usp=sharing) | [NATS-tss-v1_0-3ffb9-full](https://drive.google.com/drive/folders/17S2Xg_rVkUul4KuJdq0WaWoUuDbo8ZKB?usp=sharing) | [NATS-sss-v1_0-50262.pickle.pbz2](https://drive.google.com/file/d/1IabIvzWeDdDAWICBzFtTCMXxYWPIOIOX/view?usp=sharing) | [NATS-sss-v1_0-50262-simple.tar](https://drive.google.com/file/d/1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA/view?usp=sharing) | [NATS-sss-v1_0-50262-full](https://drive.google.com/drive/folders/1xutPQJ4bHoUV1EMArsPD0c1bUqvtMuYY?usp=sharing) |
| 2021.04.22 (Baidu-Pan) | [NATS-tss-v1_0-3ffb9.pickle.pbz2](https://pan.baidu.com/s/10z20F5s2RRPzGwRO40fLTw) (code: 8duj) | [NATS-tss-v1_0-3ffb9-simple.tar](https://pan.baidu.com/s/1vOnrHLxCB4y8cxUDrHUYAg) (code: tu1e) | [NATS-tss-v1_0-3ffb9-full](https://pan.baidu.com/s/1qbPNlI8Y1I29qMdxTo_ycA) (code:ssub) | [NATS-sss-v1_0-50262.pickle.pbz2](https://pan.baidu.com/s/1M1UaXr6y1D_RqEYg95YJcA) (code: za2h) | [NATS-sss-v1_0-50262-simple.tar](https://pan.baidu.com/s/1ek-b89Pw2qdm9MP6KKkErA) (code: e4t9) | [NATS-sss-v1_0-50262-full](https://pan.baidu.com/s/1bIruQd9pPeyArej_wttg_A) (code: htif) |

These benchmark files (without pretrained weights) can also be downloaded from [Dropbox](https://www.dropbox.com/sh/ceeo70u1buow681/AAC2M-SbKOxiIqpB0UCgXNxja?dl=0), [OneDrive](https://1drv.ms/u/s!Aqkc27lrowWDf6SvuIkSXx0UQaI?e=nfvM5r) or [Baidu-Pan (extract code: h6pm)](https://pan.baidu.com/s/144VC2BDm6iXbAVzMUpqO7A).

For the full checkpoints in `NATS-*ss-*-full`, we split the file into multiple parts (`NATS-*ss-*-full.tara*`) since they are too large to upload.
Each file is about `30GB`. For Baidu Pan, since they restrict the maximum size of each file, we further split `NATS-*ss-*-full.tara*` into `NATS-*ss-*-full.tara*-aa` and `NATS-*ss-*-full.tara*-ab`.
All splits are created by the command `split`.

**Note:** if you encounter the `quota exceed erros` when download from Google Drive, please try to (1) login your personal Google account, (2) right-click-copy the files to your personal Google Drive, and (3) download from your personal Google Drive.

## Usage

See more examples at [notebooks](notebooks).

#### 1, create the benchmark instance:
```
from nats_bench import create
# Create the API instance for the size search space in NATS
api = create(None, 'sss', fast_mode=True, verbose=True)

# Create the API instance for the topology search space in NATS
api = create(None, 'tss', fast_mode=True, verbose=True)
```

#### 2, query the performance:
```
# Query the loss / accuracy / time for 1234-th candidate architecture on CIFAR-10
# info is a dict, where you can easily figure out the meaning by key
info = api.get_more_info(1234, 'cifar10')

# Query the flops, params, latency. info is a dict.
info = api.get_cost_info(12, 'cifar10')

# Simulate the training of the 1224-th candidate:
validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(1224, dataset='cifar10', hp='12')
```

#### 3, create the instance of an architecture candidate in `NATS-Bench`:
```
# Create the instance of th 12-th candidate for CIFAR-10.
# To keep NATS-Bench repo concise, we did not include any model-related codes here because they rely on PyTorch.
# The package of [models] is defined at https://github.com/D-X-Y/AutoDL-Projects
#   so that one need to first import this package.
import xautodl
from xautodl.models import get_cell_based_tiny_net
config = api.get_net_config(12, 'cifar10')
network = get_cell_based_tiny_net(config)

# Load the pre-trained weights: params is a dict, where the key is the seed and value is the weights.
params = api.get_net_param(12, 'cifar10', None)
network.load_state_dict(next(iter(params.values())))
```

#### 4, others:
```
# Clear the parameters of the 12-th candidate.
api.clear_params(12)

# Reload all information of the 12-th candidate.
api.reload(index=12)

```

Please see [`api_test.py`](https://github.com/D-X-Y/NATS-Bench/blob/main/tests/api_test.py) for more examples.
```
from nats_bench import api_test
api_test.test_nats_bench_tss('NATS-tss-v1_0-3ffb9-simple')
api_test.test_nats_bench_tss('NATS-sss-v1_0-50262-simple')
```



## How to Re-create NATS-Bench from Scratch

**You need to use the [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects) repo to re-create NATS-Bench from scratch.**

### The Size Search Space

The following command will train all architecture candidate in the size search space with 90 epochs and use the random seed of `777`.
If you want to use a different number of training epochs, please replace `90` with it, such as `01` or `12`.
```
bash ./scripts/NATS-Bench/train-shapes.sh 00000-32767 90 777
```
The checkpoint of all candidates are located at `output/NATS-Bench-size` by default.

After training these candidate architectures, please use the following command to re-organize all checkpoints into the official benchmark file.
```
python exps/NATS-Bench/sss-collect.py
```

### The Topology Search Space

The following command will train all architecture candidate in the topology search space with 200 epochs and use the random seed of `777`/`888`/`999`.
If you want to use a different number of training epochs, please replace `200` with it, such as `12`.
```
bash scripts/NATS-Bench/train-topology.sh 00000-15624 200 '777 888 999'
```
The checkpoint of all candidates are located at `output/NATS-Bench-topology` by default.

After training these candidate architectures, please use the following command to re-organize all checkpoints into the official benchmark file.
```
python exps/NATS-Bench/tss-collect.py
```


## To Reproduce 13 Baseline NAS Algorithms in NATS-Bench

**You need to use the [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects) repo to run 13 baseline NAS methods.** Here are a brief introduction on how to run each algorithm ([NATS-algos](https://github.com/D-X-Y/AutoDL-Projects/tree/main/exps/NATS-algos)).

### Reproduce NAS methods on the topology search space

Please use the following commands to run different NAS methods on the topology search space:
```
Four multi-trial based methods:
python ./exps/NATS-algos/reinforce.py       --dataset cifar100 --search_space tss --learning_rate 0.01
python ./exps/NATS-algos/regularized_ea.py  --dataset cifar100 --search_space tss --ea_cycles 200 --ea_population 10 --ea_sample_size 3
python ./exps/NATS-algos/random_wo_share.py --dataset cifar100 --search_space tss
python ./exps/NATS-algos/bohb.py            --dataset cifar100 --search_space tss --num_samples 4 --random_fraction 0.0 --bandwidth_factor 3

DARTS (first order):
python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v1
python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v1
python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v1

DARTS (second order):
python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo darts-v2
python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo darts-v2
python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo darts-v2

GDAS:
python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo gdas
python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo gdas
python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16

SETN:
python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo setn
python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo setn
python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo setn

Random Search with Weight Sharing:
python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo random
python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo random
python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo random

ENAS:
python ./exps/NATS-algos/search-cell.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001
python ./exps/NATS-algos/search-cell.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001
python ./exps/NATS-algos/search-cell.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo enas --arch_weight_decay 0 --arch_learning_rate 0.001 --arch_eps 0.001
```

### Reproduce NAS methods on the size search space

Please use the following commands to run different NAS methods on the size search space:
```
Four multi-trial based methods:
python ./exps/NATS-algos/reinforce.py       --dataset cifar100 --search_space sss --learning_rate 0.01
python ./exps/NATS-algos/regularized_ea.py  --dataset cifar100 --search_space sss --ea_cycles 200 --ea_population 10 --ea_sample_size 3
python ./exps/NATS-algos/random_wo_share.py --dataset cifar100 --search_space sss
python ./exps/NATS-algos/bohb.py            --dataset cifar100 --search_space sss --num_samples 4 --random_fraction 0.0 --bandwidth_factor 3


Run Transformable Architecture Search (TAS), proposed in Network Pruning via Transformable Architecture Search, NeurIPS 2019

python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo tas --rand_seed 777
python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo tas --rand_seed 777
python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo tas --rand_seed 777


Run the channel search strategy in FBNet-V2 -- masking + Gumbel-Softmax :

python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo mask_gumbel --rand_seed 777
python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo mask_gumbel --rand_seed 777
python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo mask_gumbel --rand_seed 777


Run the channel search strategy in TuNAS -- masking + sampling :

python ./exps/NATS-algos/search-size.py --dataset cifar10  --data_path $TORCH_HOME/cifar.python --algo mask_rl --arch_weight_decay 0 --rand_seed 777 --use_api 0
python ./exps/NATS-algos/search-size.py --dataset cifar100 --data_path $TORCH_HOME/cifar.python --algo mask_rl --arch_weight_decay 0 --rand_seed 777
python ./exps/NATS-algos/search-size.py --dataset ImageNet16-120 --data_path $TORCH_HOME/cifar.python/ImageNet16 --algo mask_rl --arch_weight_decay 0 --rand_seed 777
```

### Final Discovered Architectures for Each Algorithm

The architecture index can be found by use `api.query_index_by_arch(architecture_string)`.

The final discovered architecture ID on CIFAR-10:
```
DARTS (first order):
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|

DARTS (second order):
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
|skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|

GDAS:
|nor_conv_3x3~0|+|nor_conv_3x3~0|none~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_3x3~2|
|nor_conv_3x3~0|+|nor_conv_3x3~0|none~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|
|avg_pool_3x3~0|+|nor_conv_3x3~0|skip_connect~1|+|nor_conv_3x3~0|nor_conv_1x1~1|nor_conv_1x1~2|
```

The final discovered architecture ID on CIFAR-100:
```
DARTS (V1):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|none~2|
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|none~2|
|skip_connect~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|nor_conv_3x3~2|

DARTS (V2):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|nor_conv_1x1~1|skip_connect~2|
|skip_connect~0|+|nor_conv_3x3~0|none~1|+|skip_connect~0|none~1|none~2|
|skip_connect~0|+|nor_conv_1x1~0|none~1|+|nor_conv_3x3~0|skip_connect~1|none~2|

GDAS:
|nor_conv_3x3~0|+|nor_conv_1x1~0|none~1|+|avg_pool_3x3~0|nor_conv_3x3~1|nor_conv_3x3~2|
|avg_pool_3x3~0|+|nor_conv_1x1~0|none~1|+|nor_conv_3x3~0|avg_pool_3x3~1|nor_conv_1x1~2|
|avg_pool_3x3~0|+|nor_conv_3x3~0|none~1|+|nor_conv_3x3~0|nor_conv_1x1~1|nor_conv_1x1~2|
```

The final discovered architecture ID on ImageNet-16-120:
```
DARTS (V1):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|nor_conv_3x3~2|
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|nor_conv_3x3~2|
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|nor_conv_1x1~2|

DARTS (V2):
|none~0|+|skip_connect~0|none~1|+|skip_connect~0|none~1|skip_connect~2|

GDAS:
|none~0|+|none~0|none~1|+|nor_conv_3x3~0|none~1|none~2|
|none~0|+|none~0|none~1|+|nor_conv_3x3~0|none~1|none~2|
|none~0|+|none~0|none~1|+|nor_conv_3x3~0|none~1|none~2|
```

## Others

We use [`black`](https://github.com/psf/black) for Python code formatter.
Please use `black . -l 120`.

## Citation

If you find that NATS-Bench helps your research, please consider citing it:
```
@article{dong2021nats,
  title   = {{NATS-Bench}: Benchmarking NAS Algorithms for Architecture Topology and Size},
  author  = {Dong, Xuanyi and Liu, Lu and Musial, Katarzyna and Gabrys, Bogdan},
  doi     = {10.1109/TPAMI.2021.3054824},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year    = {2021},
  note    = {\mbox{doi}:\url{10.1109/TPAMI.2021.3054824}}
}
@inproceedings{dong2020nasbench201,
  title     = {{NAS-Bench-201}: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
```
