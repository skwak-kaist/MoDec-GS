# 🎄 MoDec-GS

This repository is the official code for: 
> __MoDec-GS: Global-to-Local Motion Decomposition and Temporal Interval Adjustment for Compact Dynamic 3D Gaussian Splatting__
>
> Sangwoon Kwak, Joonsoo Kim, Jun Young Jeong, Won-Sik Cheong, Jihyong Oh†, Munchurl Kim† <br/>
> <span style="font-size:10px">†Co-corresponding authors</span> 
>
> ETRI, KAIST, Chung-Ang University

🏠 [Project page](https://kaist-viclab.github.io/MoDecGS-site/)

📖 [ArXiv](https://arxiv.org/abs/2501.03714)

🎥 [Demo video](https://www.youtube.com/watch?v=5L6gzc5-cw8)

## News

📌 **[2025.04.28]** We initially release the code for MoDec-GS.

📌 **[2025.02.27]** Accepted to [CVPR2025](https://cvpr.thecvf.com/).

📌 **[2025.01.07]** Draft paper was uploaded on [ArXiv](https://arxiv.org/abs/2501.03714)



## Environmental Setups

1. **Installation**

```bash
git clone https://github.com/skwak-kaist/MoDec-GS.git
cd MoDec-GS
bash install.sh
```

The install script by default creates a conda environment named **modecgs**. Our environment was test with python=3.7 and torch=1.13+cu116, but not limited to these versions. If the install script does not work properly in your environment, please check  the `setup_modec_env.sh`

2. **data preparation**

We currently provide configurations and running scripts for the [HyperNeRF](https://hypernerf.github.io/), [Dycheck-iPhone](https://github.com/KAIR-BAIR/dycheck?tab=readme-ov-file), [Nvidia-monocular](https://github.com/coltonstearns/dynamic-gaussian-marbles?tab=readme-ov-file), [PanopticSports](http://domedb.perception.cs.cmu.edu/), [D-NeRF](https://github.com/albertpumarola/D-NeRF) datasets. The datasets needs to be placed as follows: 

```
├── data
│   | dnerf 
│     ├── mutant
│     ├── standup 
│     ├── ...
│   | hypernerf
│     ├── interp
│       ├── interp_aleks-teapot
│         ├── aleks-teapot
│           ├── camera
│     ├── misc
│     ├── virg
│   | dycheck
│     ├── apple
│       ├── camera
│       ├── colmap
│     ├── ...
│   | nvidia
│     ├── Balloon1
│       ├── dense
│         ├── images
│         ├── sparse
│         ├── ...
│     ├── ...
│   | panoptic_sports
│     ├── basketball
│       ├── ims
│       ├── ...
```

Additionally, you can run other dataset by using custom configuration. The dataset path is configured by `run_scripts/dataset_config/${dataset_name}.sh`

The COLMAP data generation process refers to the code from [4DGS](https://github.com/hustvl/4DGaussians). As shown in the running scripts, it operates in the follow form: 

```bash
bash colmap.sh data/${dataset}/${scene_path} ${dataset_type}
```

The supported dataset types are blender, hypernerf, llff, nvidia, and dycheck. For additional details, you can directly refer to the [4DGS](https://github.com/hustvl/4DGaussians) repository. 

3. **running script**

The pre-written running scripts are located in the `run_scripts` folder. You can specify the GPU id, port number, and a config number when running a script. If you want to modify the configuration, place a modified file named `argument/${dataset_name}/config_${your_number}.py` and enter the corresponding number when running scripts. 



## Running

The training, rendering, evaluation processes can be executed within the pre-written running script. To run separately, you can use the following commands. 

1. **Training**

```
python train.py -s data/dycheck/apple --port 18280 --expname "dycheck_1.0/apple" --configs arguments/dycheck/config_1.0.py 
```

2. **Rendering**

```
python render.py --model_path "output/dycheck_1.0/apple" --skip_train --configs arguments/dycheck/config_1.0.py 
```

you can set the rendering option by using `--skip_train` , `--skip_test`, `--skip_video` , `--canonical_frame_render`. 

3. **Evaluation**

```
python metric.py --model_path "output/dycheck_1.0/apple"
```

For aggregating performance results by dataset, you can use `collect_metric.py`. It gathers the performance results for the pre-defined dataset and outputs them into a text file. 

```
python collect_metric.py --output_path "output/dycheck_1.0" --dataset dycheck
```

Please note that for Dycheck dataset, **masked metrics** (mPSNR, mSSIM, mLPIPS) were used as reported in the paper, and functions for the metrics  are not included in this repository. You can refer Dycheck's own modules and environments in [here](https://github.com/KAIR-BAIR/dycheck). 



## Acknowledgement

* This source code is built upon prior excellent works ([3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [4DGS](https://github.com/hustvl/4DGaussians), [Scaffold-GS](https://github.com/city-super/Scaffold-GS)). We sincerely appreciate the authors for their outstanding contributions. 

## Citation

```
@misc{kwak2025modecgsglobaltolocalmotiondecomposition,
  title={MoDec-GS: Global-to-Local Motion Decomposition and Temporal Interval Adjustment for Compact Dynamic 3D Gaussian Splatting}, 
  author={Sangwoon Kwak and Joonsoo Kim and Jun Young Jeong and Won-Sik Cheong and Jihyong Oh and Munchurl Kim},
  year={2025},
  eprint={2501.03714},
  archivePrefix={arXiv},
}
```
