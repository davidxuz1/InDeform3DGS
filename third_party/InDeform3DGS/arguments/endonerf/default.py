ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10,
    use_inpainting = True
)

DatasetParams = dict(
    # img_width = 224,
    # img_height = 224, 
    img_width = 640, # si endonerf
    img_height = 512,
    downsample = 1.0,
    test_every = 8,
    #dataset_dir = 'data/own_video'
    dataset_dir = 'data/endonerf/pulling'
)

OptimizationParams = dict(
    coarse_iterations = 0,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    iterations = 3000,
    percent_dense = 0.01,
    opacity_reset_interval = 3000,
    position_lr_max_steps = 4000,
    prune_interval = 3000,
    lambda_similarity = 10.0,
    lambda_depth_inpaint = 0.0,
    lambda_illumination = 0.00,
    lambda_diversity = 0.0,
    lambda_edge_smoothing = 0.0,
    lambda_time_consistency = 0.00,
    lambda_noise = 0.00
)

ModelHiddenParams = dict(
    curve_num = 17,
    ch_num = 10,
    init_param = 0.01,
)

# ModelParams = dict(
#     extra_mark = 'endonerf',
#     camera_extent = 10,
#     use_inpainting = True  # Nuevo parámetro
# )

# OptimizationParams = dict(
#     coarse_iterations = 0,
#     deformation_lr_init = 0.00016,
#     deformation_lr_final = 0.0000016,
#     deformation_lr_delay_mult = 0.01,
#     iterations = 3000,
#     percent_dense = 0.01,
#     opacity_reset_interval = 3000,
#     position_lr_max_steps = 4000,
#     prune_interval = 3000,
#     # Nuevos parámetros para inpainting
#     lambda_similarity = 0.01,
#     lambda_depth_inpaint = 0.01,
#     lambda_illumination = 0.01,
#     lambda_diversity = 0.01,
#     lambda_edge_smoothing = 0.01,
#     lambda_time_consistency = 0.01,
#     lambda_noise = 0.01
# )

# ModelHiddenParams = dict(
#     curve_num = 17, # number of learnable basis functions. This number was set to 17 for all the experiments in paper (https://arxiv.org/abs/2405.17835)

#     ch_num = 10, # channel number of deformable attributes: 10 = 3 (scale) + 3 (mean) + 4 (rotation)
#     init_param = 0.01, )

