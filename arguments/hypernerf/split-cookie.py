_base_="./base.py"
ModelParams=dict(
    kplanes_config_local = { 
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
	multires_local = [1,2,4],
)

OptimizationParams = dict(
    iterations = 14000,
    min_opacity = 0.005,
    densify_grad_threshold = 0.005,
    )
