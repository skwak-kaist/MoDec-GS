#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        
        ############################
        # Scaffold-GS related params
        ############################
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.01 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4
        self.use_feat_bank = False
        self.lod = 0

        self.appearance_dim = 16
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        
        ############################
        # 4DGS-related params
        ############################
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.render_process=False
        self.add_points=False
        self.extension=".png"
        self.llffhold=8

        ############################
        # MoDec-GS related params
        ############################      
        self.anchor_dynamics = True # Indicate whether to use anchor dynamics
        self.anchor_dynamics_share = True # Indicate whether to share the anchor dynamics (If true, dG and dL are shared)

        # deformation parameters
        self.GLMD = True
        # deformation method
        self.GAD_method = "implicit"
        self.LGD_method = "explicit"
        
        # temporal adjustment parameters (Sec 4.4)
        self.TIA = True
        self.TIA_num_segments = 8
        self.TIA_step_size = 0.01
        self.TIA_threshold = 1.5

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.opacity_activation = False # MoDec-GS paramter, default 0. 
        super().__init__(parser, "Pipeline Parameters")
        
class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width =128 # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4 # useless
        self.defor_depth = 1 # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10 # useless
        self.scale_rotation_pe = 2 # useless
        self.opacity_pe = 2 # useless
        self.timenet_width = 64 # useless
        self.timenet_output = 32 # useless
        self.bounds = 1.6 
        
        self.plane_tv_weight_global = 0.0001 # TV loss of spatial grid (global)
        self.time_smoothness_weight_global = 0.0005 # TV loss of temporal grid (global)
        self.l1_time_planes_global = 0.00005 # TV loss of temporal grid (global)
        
        self.plane_tv_weight = 0.0002 # TV loss of spatial grid (local), if there is no GLMD, this value will be used for the defomation
        self.time_smoothness_weight = 0.001 # TV loss of temporal grid
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.kplanes_config_global = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 16,
                             'resolution': [32, 32, 32, 10]  
                            }
        self.multires_global = [1, 2] # multi resolution of voxel grid
        
        self.kplanes_config_local = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 16,
                             'resolution': [64, 64, 64, 100]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }
        self.multires_local = [1, 2] # multi resolution of voxel grid
        
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_ds=False # cancel the deformation of Gaussians' scaling
        self.no_dr=False # cancel the deformation of Gaussians' rotations
        self.no_do=True # cancel the deformation of Gaussians' opacity
        self.no_dshs=True # cancel the deformation of SH colors.
        self.empty_voxel=False # useless
        self.grid_pe=0 # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp=False # useless
        self.apply_rotation=False # useless

        ############################
        # MoDec-GS related params
        ############################                  
        # global dynamics in GAD (Sec 4.2)
        self.anchor_dynamics_position = True
        self.anchor_dynamics_global_type = "mask" # mask or mul        
        
        # local dynamics in GAD (Sec 4.2)
        self.anchor_dynamics_local_context_feature = True
        self.anchor_dynamics_offset = True
        self.anchor_dynamics_scaling = True
        self.anchor_dynamics_local_type = "mul" # mask or mul       
        # dynamics activation
        self.dynamics_activation='sigmoid' # relu or leakyrelu or sigmoid
        
        # Deforamtion
        # deformed attributes
        self.anchor_deform = True
        self.local_context_feature_deform = True
        self.grid_offsets_deform = True
        self.grid_scale_deform = True
        
        self.deform_feat_dim = None # it will be set in the model (feat_dim of ModelParams)
        self.deform_n_offsets = None # it will be set in the model (n_offsets of ModelParams)

        super().__init__(parser, "ModelHiddenParams")

        
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        ############################
        # 4DGS-related params
        ############################
        self.dataloader=False
        self.zerostamp_init=False
        self.custom_sampler=None
        
        self.coarse_iterations = 3000
        #self.position_lr_init = 0.00016
        #self.position_lr_final = 0.0000016
        #self.position_lr_delay_mult = 0.01
        #self.position_lr_max_steps = 20_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        # self.feature_lr = 0.0025
        # self.opacity_lr = 0.05
        # self.scaling_lr = 0.005
        # self.rotation_lr = 0.001
        self.percent_dense = 0.01
        #self.lambda_dssim = 0
        self.lambda_lpips = 0
        self.weight_constraint_init= 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        
        self.batch_size=4
        self.add_point=False

        ############################
        # Scaffold-GS-related params
        ############################
        self.iterations = 20_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        ############################
        # MoDec-GS related params
        ############################     
        # dynamic masks
        self.dynamics_lr_init = 0.002
        self.dynamics_lr_final = 0.00002
        self.dynamics_lr_delay_mult = 0.01
        self.dynamics_lr_max_steps = 30_000
        self.lambda_dynamics = 0.0001
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.01
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        # for temporal adjustment
        self.temporal_adjustment_until = 45_000
        self.temporal_adjustment_from = 5000
        self.temporal_adjustment_interval = 1000
            
        # for dynamics
        self.dynamics_loss = None # None, 

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
