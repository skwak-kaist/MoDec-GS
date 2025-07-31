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

import torch
from einops import repeat
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False):
    """
    Render function
    """
    
    # assign attributes        
    is_training = pc.get_color_mlp.training   
    anchor = pc.get_anchor 
    feat = pc._anchor_feat 
    grid_offsets = pc._offset
    grid_scaling = pc.get_scaling 
   
    # coarse stage
    if "coarse" in stage:   
        if is_training:
            means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                generate_neural_gaussians(viewpoint_camera, pc, visible_mask, anchor, feat, grid_offsets, grid_scaling, is_training=is_training, cam_type=cam_type)
        else:
            means3D_final, color, opacity_final, scales_final, rotations_final = \
                generate_neural_gaussians(viewpoint_camera, pc, visible_mask, anchor, feat, grid_offsets, grid_scaling, is_training=is_training, cam_type=cam_type)
    
    elif "fine" in stage:
        # anchor dynamics
        if pc.anchor_dynamics != 0: 
            dynamics = pc._dynamics 
        else:
            dynamics = None
                
        # time_base processing
        if cam_type != "PanopticSports":
            time_base = viewpoint_camera.time # 0~1 사이의 float 값
        else:
            time_base = viewpoint_camera['time']
        
        # get canonical time and repeat it as the length of the anchor
        canonical_time = pc.get_canonical_time(time_base)
        time = torch.tensor(time_base).to(anchor.device).repeat(anchor.shape[0],1) 
        canonical_time = torch.tensor(canonical_time).to(anchor.device).repeat(anchor.shape[0],1) 
    
        ###### Global-to-Local hierarchical deformation ######
        # Deformation stage 1
        # GAD: Global Anchor Deformation
        # Implicit deformation only
        anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed = pc._deformation_GAD(anchor, feat, grid_offsets, grid_scaling, dynamics, canonical_time)
               
        # Deformation stage 2
        # LGD: Local Gaussian Deformation
        # Explicit (default) or Implicit deformation
        if pc.LGD_method == "explicit": 
            # Neural Gaussian Generation first
            if is_training:
                means3D, color, opacity, scales, rotations, neural_opacity, mask = \
                    generate_neural_gaussians(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)
            else:
                means3D, color, opacity, scales, rotations = \
                    generate_neural_gaussians(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)     

            shs = None
            time = torch.tensor(time_base).to(means3D.device).repeat(means3D.shape[0],1)
            
            # And then deformation 
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation_LGD(means3D, scales, rotations, opacity, shs, time)
            
        elif pc.LGD_method == "implicit":
            # deformation first
            anchor_final, feat_final, grid_offsets_final, grid_scaling_final = pc._deformation_LGD(anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, dynamics, time)

            # and then neural gaussian generation
            if is_training:
                means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                    generate_neural_gaussians(viewpoint_camera, pc, visible_mask, anchor_final, feat_final, grid_offsets_final, grid_scaling_final, is_training=is_training, cam_type=cam_type)
            else:
                means3D_final, color, opacity_final, scales_final, rotations_final = \
                    generate_neural_gaussians(viewpoint_camera, pc, visible_mask, anchor_final, feat_final, grid_offsets_final, grid_scaling_final, is_training=is_training, cam_type=cam_type)
        
                                                                
    # Common path
    # screenspace point generation
    screenspace_points = torch.zeros_like(means3D_final, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Set up rasterization configuration
    if cam_type != "PanopticSports": # 
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            #sh_degree=pc.active_sh_degree,
            sh_degree=1,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
    else:
        raster_settings = viewpoint_camera['camera'] 
        
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points 
    shs_final = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python: 
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    
    # rotation activation
    rotations_final = pc.rotation_activation(rotations_final)
    
    # opacity activation (default: False)
    if pipe.opacity_activation:
        opacity_final = pc.opacity_activation(opacity_final)
    
    # RASTERIZATION    
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = color,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scales_final,
                "depth":depth,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth":depth,
                }


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, anchor = None, feat = None, grid_offsets = None, grid_scaling = None, is_training=False, cam_type=None):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    feat = feat[visible_mask]
    anchor = anchor[visible_mask]
    grid_offsets = grid_offsets[visible_mask]
    grid_scaling = grid_scaling[visible_mask]    
    
    if cam_type != "PanopticSports":
        viewpoint_camera.camera_center = viewpoint_camera.camera_center.cuda()
        ob_view = anchor - viewpoint_camera.camera_center
    else:
        ob_view = anchor - viewpoint_camera['camera'].campos.cuda()
        
    ob_dist = ob_view.norm(dim=1, keepdim=True) # dist
    ob_view = ob_view / ob_dist # view

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        #camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10 # fixing cuda error 
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist) 

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    
    #rot = pc.rotation_activation(scale_rot[:,3:7])
    rot = scale_rot[:,3:7] # activation of rotation will be done in the render function
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, cam_type=None):

    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    if cam_type != "PanopticSports": 
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        # from cpu to cuda
        viewpoint_camera.world_view_transform = viewpoint_camera.world_view_transform.cuda()
        viewpoint_camera.full_proj_transform = viewpoint_camera.full_proj_transform.cuda()
        viewpoint_camera.camera_center = viewpoint_camera.camera_center.cuda()  

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
    else:
        raster_settings = viewpoint_camera['camera']

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
    

def render_canonical(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False, canonical_times=None, still_camera=None):
    """
    variations of render function, it is used only for ablation study
    """
    
    # assign attributes        
    is_training = pc.get_color_mlp.training   
    anchor = pc.get_anchor 
    feat = pc._anchor_feat 
    grid_offsets = pc._offset
    grid_scaling = pc.get_scaling 
   
    if pc.anchor_dynamics != 0: 
        dynamics = pc._dynamics 
    else:
        dynamics = None
            
   
    if cam_type != "PanopticSports":
        time_base = viewpoint_camera.time 
    else:
        time_base = viewpoint_camera['time']
 
    
    if canonical_times == 'global': # Globa CS rendering
        # direct rendering the anchor
        if is_training:
            means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                generate_neural_gaussians(viewpoint_camera, pc, visible_mask, anchor, feat, grid_offsets, grid_scaling, is_training=is_training, cam_type=cam_type)
        else:
            means3D_final, color, opacity_final, scales_final, rotations_final = \
                generate_neural_gaussians(viewpoint_camera, pc, visible_mask, anchor, feat, grid_offsets, grid_scaling, is_training=is_training, cam_type=cam_type)
    
    elif canonical_times == 'local': # Local CS rendering 
        # GAD, but not LGD --> neural gaussian generation (moving camera)
        canonical_time = pc.get_canonical_time(time_base)
        time = torch.tensor(time_base).to(anchor.device).repeat(anchor.shape[0],1) 
        canonical_time = torch.tensor(canonical_time).to(anchor.device).repeat(anchor.shape[0],1) 
        anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed = pc._deformation_GAD(anchor, feat, grid_offsets, grid_scaling, dynamics, canonical_time)
        
        if is_training:
            means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                generate_neural_gaussians(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)
        else:
            means3D_final, color, opacity_final, scales_final, rotations_final = \
                generate_neural_gaussians(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)        
                
    elif canonical_times == 'local_still': # local canonical rendering still camera
        # GAD, but not LGD --> neural gaussian generation (still camera)
        canonical_time = pc.get_canonical_time(time_base)
        time = torch.tensor(time_base).to(anchor.device).repeat(anchor.shape[0],1) 
        canonical_time = torch.tensor(canonical_time).to(anchor.device).repeat(anchor.shape[0],1)         
        anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed = pc._deformation_GAD(anchor, feat, grid_offsets, grid_scaling, dynamics, canonical_time)
        
        if is_training:
            means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                generate_neural_gaussians(still_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)
        else:
            means3D_final, color, opacity_final, scales_final, rotations_final = \
                generate_neural_gaussians(still_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)       
                
        viewpoint_camera = still_camera
        
    elif canonical_times == 'render_still': # rendering with entire GLMD but still camera
        canonical_time = pc.get_canonical_time(time_base)       
        time = torch.tensor(time_base).to(anchor.device).repeat(anchor.shape[0],1) 
        canonical_time = torch.tensor(canonical_time).to(anchor.device).repeat(anchor.shape[0],1) 
        anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed = pc._deformation_G2C(anchor, feat, grid_offsets, grid_scaling, dynamics, canonical_time)
        
        if is_training:
            means3D, color, opacity, scales, rotations, neural_opacity, mask = \
                generate_neural_gaussians(still_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)
        else:
            means3D, color, opacity, scales, rotations = \
                generate_neural_gaussians(still_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)     

        shs = None
        time = torch.tensor(time_base).to(means3D.device).repeat(means3D.shape[0],1)
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation_C2L(means3D, scales,
                                                                rotations, opacity, shs, time) 
        viewpoint_camera = still_camera
                
    else:
        raise ValueError("Invalid canonical times")
                                                                 
    # Common path
    # screenspace point generation
    screenspace_points = torch.zeros_like(means3D_final, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Set up rasterization configuration
    if cam_type != "PanopticSports": # 
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            #sh_degree=pc.active_sh_degree,
            sh_degree=1,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
    else:
        raster_settings = viewpoint_camera['camera'] 
        
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points 
    shs_final = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python: 
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    
    # rotation activation
    rotations_final = pc.rotation_activation(rotations_final)
    
    # opacity activation (default: False)
    if pipe.opacity_activation:
        opacity_final = pc.opacity_activation(opacity_final)
    
    # RASTERIZATION    
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = color,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scales_final,
                "depth":depth,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth":depth,
                }


