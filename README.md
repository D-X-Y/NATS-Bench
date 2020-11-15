# [NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size](https://arxiv.org/pdf/2009.00437.pdf)

Neural architecture search (NAS) has attracted a lot of attention and has been illustrated to bring tangible benefits in a large number of applications in the past few years. Network topology and network size have been regarded as two of the most important aspects for the performance of deep learning models and the community has spawned lots of searching algorithms for both of those aspects of the neural architectures. However, the performance gain from these searching algorithms is achieved under different search spaces and training setups. This makes the overall performance of the algorithms incomparable and the improvement from a sub-module of the searching model unclear.
In this paper, we propose NATS-Bench, a unified benchmark on searching for both topology and size, for (almost) any up-to-date NAS algorithm.
NATS-Bench includes the search space of 15,625 neural cell candidates for architecture topology and 32,768 for architecture size on three datasets.
We analyze the validity of our benchmark in terms of various criteria and performance comparison of all candidates in the search space.
We also show the versatility of NATS-Bench by benchmarking 13 recent state-of-the-art NAS algorithms on it. All logs and diagnostic information trained using the same setup for each candidate are provided.
This facilitates a much larger community of researchers to focus on developing better NAS algorithms in a more comparable and computationally effective environment.

**You can use `pip install nats_bench` to install the library of NATS-Bench.**

If you are seeking how to re-create NATS-Bench from scratch or reproduce benchmarked results, please see [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects/blob/master/docs/NATS-Bench.md#how-to-re-create-nats-bench-from-scratch).

If you have questions, please ask at [here](https://github.com/D-X-Y/AutoDL-Projects/issues) or email me :)


## Preparation and Download

In NATS-Bench, we (create and) use three image datasets -- CIFAR-10, CIFAR-100, and ImageNet16-120.
For more details, please see Sec-3.2 in [the NATS-Bench paper](https://arxiv.org/pdf/2009.00437.pdf). To download these three datasets, please find them at [Google Drive](https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4?usp=sharing).
To create the `ImageNet16-120` PyTorch dataset, please call [AutoDL-Projects/lib/datasets/ImageNet16](https://github.com/D-X-Y/AutoDL-Projects/blob/master/lib/datasets/get_dataset_with_transform.py#L168), by using:
```
train_data = ImageNet16(root, True , train_transform, 120)
test_data  = ImageNet16(root, False, test_transform , 120)
```

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
| 2020.08.31 | [NATS-tss-v1_0-3ffb9.pickle.pbz2](https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing) | [NATS-tss-v1_0-3ffb9-simple.tar](https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view?usp=sharing) | [NATS-tss-v1_0-3ffb9-full](https://drive.google.com/drive/folders/17S2Xg_rVkUul4KuJdq0WaWoUuDbo8ZKB?usp=sharing) | [NATS-sss-v1_0-50262.pickle.pbz2](https://drive.google.com/file/d/1IabIvzWeDdDAWICBzFtTCMXxYWPIOIOX/view?usp=sharing) | [NATS-sss-v1_0-50262-simple.tar](https://drive.google.com/file/d/1scOMTUwcQhAMa_IMedp9lTzwmgqHLGgA/view?usp=sharing) | [NATS-sss-v1_0-50262-full](api.reload(index=12)) |

These benchmark files (without pretrained weights) can also be downloaded from [Dropbox](https://www.dropbox.com/sh/ceeo70u1buow681/AAC2M-SbKOxiIqpB0UCgXNxja?dl=0), [OneDrive](https://1drv.ms/u/s!Aqkc27lrowWDf6SvuIkSXx0UQaI?e=nfvM5r) or [Baidu-Pan (extract code: h6pm)](https://pan.baidu.com/s/144VC2BDm6iXbAVzMUpqO7A).

## Usage

1, create the benchmark instance:
```
from nats_bench import create
# Create the API instance for the size search space in NATS
api = create(None, 'sss', fast_mode=True, verbose=True)

# Create the API instance for the topology search space in NATS
api = create(None, 'tss', fast_mode=True, verbose=True)
```

2, query the performance:
```
# Query the loss / accuracy / time for 1234-th candidate architecture on CIFAR-10
# info is a dict, where you can easily figure out the meaning by key
info = api.get_more_info(1234, 'cifar10')

# Query the flops, params, latency. info is a dict.
info = api.get_cost_info(12, 'cifar10')

# Simulate the training of the 1224-th candidate:
validation_accuracy, latency, time_cost, current_total_time_cost = api.simulate_train_eval(1224, dataset='cifar10', hp='12')
```

3, others:
```
# Clear the parameters of the 12-th candidate.
api.clear_params(12)

# Reload all information of the 12-th candidate.
api.reload(index=12)

# Create the instance of th 12-th candidate for CIFAR-10.
# To keep NATS-Bench repo concise, we did not include any model-related codes here because they rely on PyTorch.
# The package of [models] is defined at https://github.com/D-X-Y/AutoDL-Projects
from models import get_cell_based_tiny_net
config = api.get_net_config(12, 'cifar10')
network = get_cell_based_tiny_net(config)

# Load the pre-trained weights: params is a dict, where the key is the seed and value is the weights.
params = api.get_net_param(12, 'cifar10', None)
network.load_state_dict(next(iter(params.values())))
```


Please see [`api_test.py`](https://github.com/D-X-Y/NATS-Bench/blob/main/nats_bench/api_test.py) for more examples.
```
from nats_bench import api_test
api_test.test_nats_bench_tss('NATS-tss-v1_0-3ffb9-simple')
api_test.test_nats_bench_tss('NATS-sss-v1_0-50262-simple')
```


## Citation

If you find that NATS-Bench helps your research, please consider citing it:
```
@article{dong2020nats,
  title={{NATS-Bench}: Benchmarking NAS Algorithms for Architecture Topology and Size},
  author={Dong, Xuanyi and Liu, Lu and Musial, Katarzyna and Gabrys, Bogdan},
  journal={arXiv preprint arXiv:2009.00437},
  year={2020}
}
```
