
ModelParams = dict(
    n_offsets = 10, 
    voxel_size = 0.01, 
    feat_dim = 32, 
    
	# anchor dynamics
	anchor_dynamics = True, 
	anchor_dynamics_share = False, 

	# TIA
	TIA_num_segments = 8,
	TIA_step_size = 0.01,
	TIA_threshold = 1.5,
)

ModelHiddenParams = dict(
	# global kplanes
	kplanes_config_global = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [32, 32, 32, 10]
    },
    multires_global = [1,2],
    # local kplanes
    kplanes_config_local = { 
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 100]
    },
    multires_local = [1,2],
        
    anchor_deform = True, 
    local_context_feature_deform = True,
    grid_offsets_deform = True,
    grid_scale_deform = True,
    
	# regularizer for global kplanes
    plane_tv_weight_global = 0.0001,
    time_smoothness_weight_global = 0.0005,
    l1_time_planes_global =  0.00005,
	# regularizer for local kplanes
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
	
    render_process = True, 
    
    # dynamics
    anchor_dynamics_position = True, # global dynamics
    anchor_dyamics_global_type = "mul", # mask or mul
    
    anchor_dynamics_local_context_feature = True, # local dynamics
    anchor_dynamics_offset = True,
    anchor_dynamics_scaling = True,
    anchor_dynamics_local_type = "mul", # mask or mul       
)


OptimizationParams = dict(
    # dataloader=True,
    iterations = 30000,
    coarse_iterations = 3000,
    batch_size = 4,
   
    dynamics_loss = None,
    lambda_dynamics = 0.0001,
    
    min_opacity = 0.01,
    success_threshold = 0.7, 
    densify_grad_threshold = 0.0002,
            
)
