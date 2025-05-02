_base_="./base.py"
ModelParams=dict(
	multires_local = [1,2,4],
)

OptimizationParams = dict(
    # dataloader=True,
    iterations = 14000,
    batch_size = 2,
    densify_grad_threshold = 0.005,
    min_opacity = 0.005,
    )
