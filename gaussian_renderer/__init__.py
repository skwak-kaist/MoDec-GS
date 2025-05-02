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



def render_old(gvc_params, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False, canonical_times=None):

    # common path of render function
    # GVC testmode에 따라서 render 함수를 호출하는 방식이 달라짐

    is_training = pc.get_color_mlp.training

    if gvc_params["GVC_testmode"] == 0:
        if is_training:
            rendered_image, screenspace_points, radii, mask, neural_opacity, scaling, depth = render_original(
                viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
        else:
            rendered_image, screenspace_points, radii, depth = render_original(
                viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
            
    elif gvc_params["GVC_testmode"] == 1:
        # initial frame: scaffold-GS, deformation: Gaussian attributes by 4DGS
        if is_training:
            rendered_image, screenspace_points, radii, mask, neural_opacity, scaling, depth = render_test1(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
        else:
            rendered_image, screenspace_points, radii, depth = render_test1(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
    elif gvc_params["GVC_testmode"] == 2:
        # inifial frame: scaffold-GS, deformation: scaffold-GS features
        if is_training:
            rendered_image, screenspace_points, radii, mask, neural_opacity, scaling, depth = render_test2(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
        else:
            rendered_image, screenspace_points, radii, depth = render_test2(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
    elif gvc_params["GVC_testmode"] == 3:
        # GVC_testmode 2 + dynamics attributes
        if is_training:
            rendered_image, screenspace_points, radii, mask, neural_opacity, scaling, depth = render_test3(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
        else:
            rendered_image, screenspace_points, radii, depth = render_test3(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)

    elif gvc_params["GVC_testmode"] == 4:
        # Feature deformation + learnable dynamics + temporal scaffolding
        if is_training:
            rendered_image, screenspace_points, radii, mask, neural_opacity, scaling, depth = render_test4(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
        else:
            rendered_image, screenspace_points, radii, depth = render_test4(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)

    elif gvc_params["GVC_testmode"] == 5:
        # Feature deformation + learnable dynamics + temporal scaffolding + temporal adjustment
        if is_training:
            rendered_image, screenspace_points, radii, mask, neural_opacity, scaling, depth = render_test5(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
        else:
            rendered_image, screenspace_points, radii, depth = render_test5(
                gvc_params, viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, stage, cam_type, visible_mask, retain_grad)
    
    
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "depth":depth,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth":depth,
                }


def generate_neural_gaussians_v0(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    # visible_mask에 True가 몇 개인지 확인
    # print("visible_mask:",visible_mask.sum()) # 너무 큼

    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    #print("anchor device:",anchor.device) # cuda:0
    #print("viewpoint_camera.camera_center device:",viewpoint_camera.camera_center.device) # cpu
    viewpoint_camera.camera_center = viewpoint_camera.camera_center.cuda()
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

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
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10 # 왜 인지는 모르겠지만 이렇게 하니까 cuda error가 발생하지 않음
            
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        # print("cat_local_view_wodist:",cat_local_view_wodist.shape) # 얘는 죄가 없음
        #print("camera_indicies:",camera_indicies) # camera_indicies: torch.Size([10562])
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist) 
        # 여기서 계속 CUDA error 발생함
        # CUBLAS_STATUS_EXECUTION_FAILED --> dimension mistmatch이슈가 많다고 함
        # cat_local_view_wodist: shape torch.size([10562, 35])


    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)
    #print("mask:",mask.shape) # mask: torch.Size([105620])
    #print("neural_opacity:",neural_opacity.shape) # neural_opacity: torch.Size([105620, 1])

    # select opacity 
    opacity = neural_opacity[mask]
    # neural_opacity: 105620, 1
    # mask: 105620

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
    
    # modified!!! - s.kwak
    # v0 조차 rotation에 대해서는 activation function을 적용하지 않음
    #rot = pc.rotation_activation(scale_rot[:,3:7])
    rot = scale_rot[:,3:7]
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def generate_neural_gaussians_v1(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    # visible_mask에 True가 몇 개인지 확인
    # print("visible_mask:",visible_mask.sum()) # 너무 큼

    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    #print("anchor device:",anchor.device) # cuda:0
    #print("viewpoint_camera.camera_center device:",viewpoint_camera.camera_center.device) # cpu
    viewpoint_camera.camera_center = viewpoint_camera.camera_center.cuda()
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

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
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10 # 왜 인지는 모르겠지만 이렇게 하니까 cuda error가 발생하지 않음
            
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        # print("cat_local_view_wodist:",cat_local_view_wodist.shape) # 얘는 죄가 없음
        #print("camera_indicies:",camera_indicies) # camera_indicies: torch.Size([10562])
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist) 
        # 여기서 계속 CUDA error 발생함
        # CUBLAS_STATUS_EXECUTION_FAILED --> dimension mistmatch이슈가 많다고 함
        # cat_local_view_wodist: shape torch.size([10562, 35])


    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)
    #print("mask:",mask.shape) # mask: torch.Size([105620])
    #print("neural_opacity:",neural_opacity.shape) # neural_opacity: torch.Size([105620, 1])

    # select opacity 
    opacity = neural_opacity[mask]
    # neural_opacity: 105620, 1
    # mask: 105620

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
    # modified !!! - s.kwak
    # scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    
    # modified!!! - s.kwak
    #rot = pc.rotation_activation(scale_rot[:,3:7])
    rot = scale_rot[:,3:7]
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling_repeat[:,3:], scale_rot[:,:3], rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling_repeat[:,3:], scale_rot[:,:3], rot



def generate_neural_gaussians_v2(viewpoint_camera, pc : GaussianModel, visible_mask=None, anchor = None, 
                                 feat = None, grid_offsets = None, grid_scaling = None, is_training=False, cam_type=None):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    # visible_mask에 True가 몇 개인지 확인
    # print("visible_mask:",visible_mask.sum()) # 너무 큼

    # deformation 해주기 위해 feat과 anchor를 외부에서 입력해주는 버전
    #feat = pc._anchor_feat[visible_mask]
    #anchor = pc.get_anchor[visible_mask]
    #grid_offsets = pc._offset[visible_mask]
    #grid_scaling = pc.get_scaling[visible_mask]
    
    feat = feat[visible_mask]
    anchor = anchor[visible_mask]
    grid_offsets = grid_offsets[visible_mask]
    grid_scaling = grid_scaling[visible_mask]
    
    ## get view properties for anchor
    #print("anchor device:",anchor.device) # cuda:0
    #print("viewpoint_camera.camera_center device:",viewpoint_camera.camera_center.device) # cpu
    
    print("viewpoint_camera:",viewpoint_camera)
    
    # cam_type이 PanopticSports가 아닌 경우
    if cam_type != "PanopticSports":
        viewpoint_camera.camera_center = viewpoint_camera.camera_center.cuda()
        ob_view = anchor - viewpoint_camera.camera_center
    else:
        
        #viewpoint_camera['camera'].campos = viewpoint_camera['camera'].campos.cuda()
        ob_view = anchor - viewpoint_camera['camera'].campos.cuda()
        
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

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
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10 # 왜 인지는 모르겠지만 이렇게 하니까 cuda error가 발생하지 않음
            
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        # print("cat_local_view_wodist:",cat_local_view_wodist.shape) # 얘는 죄가 없음
        #print("camera_indicies:",camera_indicies) # camera_indicies: torch.Size([10562])
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist) 
        # 여기서 계속 CUDA error 발생함
        # CUBLAS_STATUS_EXECUTION_FAILED --> dimension mistmatch이슈가 많다고 함
        # cat_local_view_wodist: shape torch.size([10562, 35])


    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)
    #print("mask:",mask.shape) # mask: torch.Size([105620])
    #print("neural_opacity:",neural_opacity.shape) # neural_opacity: torch.Size([105620, 1])

    # select opacity 
    opacity = neural_opacity[mask]
    # neural_opacity: 105620, 1
    # mask: 105620

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
    
    # modified!!! - s.kwak
    # v0 조차 rotation에 대해서는 activation function을 적용하지 않음
    #rot = pc.rotation_activation(scale_rot[:,3:7])
    rot = scale_rot[:,3:7]
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot


# GVC test mode 5일 때 호출되는 render 함수

# GVC test mode 4일 때 호출되는 render 함수
def render_test4(gvc_params, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
              override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False):
    
    is_training = pc.get_color_mlp.training   
    anchor = pc.get_anchor # ([10562, 3])
    feat = pc._anchor_feat # ([10562, 32])   
    grid_offsets = pc._offset # ([10562, 10, 3])
    grid_scaling = pc.get_scaling # ([10562, 6])
    
    if gvc_params["GVC_Dynamics"] != 0:
        dynamics = pc._dynamics # ([10562, 1]) or ([10562, 2])
    else:
        dynamics = None
    
    # time 정보 처리 부분
    if cam_type != "PanopticSports":
        time_base = viewpoint_camera.time # 0~1 사이의 float 값
    else:
        time_base = viewpoint_camera['time']
    num_segments = gvc_params["GVC_num_of_segments"] # default 2
    segment_time = 1.0 / num_segments
    
    canonical_time = (time_base // segment_time) * segment_time
    
    time = torch.tensor(time_base).to(anchor.device).repeat(anchor.shape[0],1) # time을 anchor 길이 만큼만 repeat
    canonical_time = torch.tensor(canonical_time).to(anchor.device).repeat(anchor.shape[0],1) # canonical_time을 anchor 길이 만큼만 repeat
        
    if "coarse" in stage:   
        if is_training:
            means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor, feat, grid_offsets, grid_scaling, is_training=is_training, cam_type=cam_type)
        else:
            means3D_final, color, opacity_final, scales_final, rotations_final = \
                generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor, feat, grid_offsets, grid_scaling, is_training=is_training, cam_type=cam_type)
    
    elif "fine" in stage:

        ###### Global-to-Local hierarchical deformation ######
        # Deformation 1: global to canonical
        # Feature deformation only
        anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed = pc._deformation_G2C(anchor, feat, grid_offsets, grid_scaling, dynamics, canonical_time)
               
        # Deformation 2: canonical to local
        if gvc_params["GVC_local_deform_method"] == "explicit": # explicit(gaussian) or implicit(feature)
            
            # deform된 anchor와 feat를 가지고 neural gaussian을 생성
            if is_training:
                means3D, color, opacity, scales, rotations, neural_opacity, mask = \
                    generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)
            else:
                means3D, color, opacity, scales, rotations = \
                    generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)     

            shs = None
            time = torch.tensor(time_base).to(means3D.device).repeat(means3D.shape[0],1)
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation_C2L(means3D, scales,
                                                                    rotations, opacity, shs, time)

        elif gvc_params["GVC_local_deform_method"] == "implicit":

            # anchor feature deformation을 먼저 함
            
            anchor_final, feat_final, grid_offsets_final, grid_scaling_final = pc._deformation_C2L(anchor, feat, grid_offsets, grid_scaling, dynamics, time)
            
             # deform된 anchor와 feat를 가지고 neural gaussian을 생성
            if is_training:
                means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                    generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)
            else:
                means3D_final, color, opacity_final, scales_final, rotations_final = \
                    generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)     

            #neural_opacity = opacity_final
                                                                
            
    # 이후 과정은 공통임
    # screenspace point 생성    
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
    
    if pipe.compute_cov3D_python: # False임
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    
    # 얘는 무조건 함
    rotations_final = pc.rotation_activation(rotations_final)

    # 1인경우 activation 하도록    
    if gvc_params["GVC_Opacity_Activation"] == 1:
        opacity_final = pc.opacity_activation(opacity_final)
        
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
        return rendered_image, screenspace_points, radii, mask, neural_opacity, scales_final, depth
    else:
        return rendered_image, screenspace_points, radii, depth


# GVC test mode 3일 때 호출되는 render 함수
def render_test3(gvc_params, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
              override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False):
    
    is_training = pc.get_color_mlp.training   
    anchor = pc.get_anchor # ([10562, 3])
    feat = pc._anchor_feat # ([10562, 32])   
    grid_offsets = pc._offset # ([10562, 10, 3])
    grid_scaling = pc.get_scaling # ([10562, 6])
    
    if gvc_params["GVC_Dynamics"] != 0:
        dynamics = pc._dynamics # ([10562, 1]) or ([10562, 2])
    else:
        dynamics = None
        
    if "coarse" in stage:   
        if is_training:
            means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor, feat, grid_offsets, grid_scaling, is_training=is_training, cam_type=cam_type)
        else:
            means3D_final, color, opacity_final, scales_final, rotations_final = \
                generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor, feat, grid_offsets, grid_scaling, is_training=is_training, cam_type=cam_type)
    
    elif "fine" in stage:
        # time 정보
        if cam_type != "PanopticSports":
            time = torch.tensor(viewpoint_camera.time).to(anchor.device).repeat(anchor.shape[0],1) # time을 anchor 길이 만큼만 repeat하면 됨
        else:
            time = torch.tensor(viewpoint_camera['time']).to(anchor.device).repeat(anchor.shape[0],1)

        #
        # Deformation
        anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed = pc._deformation(anchor, feat, grid_offsets, grid_scaling, dynamics, time)
        #
        #
        dynamics_final = dynamics
              
        # deform된 anchor와 feat를 가지고 neural gaussian을 생성
        if is_training:
            means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)
        else:
            means3D_final, color, opacity_final, scales_final, rotations_final = \
                generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)     

    
    # 이후 과정은 공통임
    # screenspace point 생성    
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
    
    if pipe.compute_cov3D_python: # False임
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    
    # 얘는 무조건 함
    rotations_final = pc.rotation_activation(rotations_final)

    # 1인경우 activation 하도록    
    if gvc_params["GVC_Opacity_Activation"] == 1:
        opacity_final = pc.opacity_activation(opacity_final)
        
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
        return rendered_image, screenspace_points, radii, mask, neural_opacity, scales_final, depth
    else:
        return rendered_image, screenspace_points, radii, depth



# GVC test mode 2일 때 호출되는 render 함수
def render_test2(gvc_params, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
              override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False):
    # typical
    is_training = pc.get_color_mlp.training
    
    # anchor랑 feat를 미리 꺼내놓자
    anchor = pc.get_anchor # ([10562, 3])
    feat = pc._anchor_feat # ([10562, 32])   
    grid_offsets = pc._offset # ([10562, 10, 3])
    grid_scaling = pc.get_scaling # ([10562, 6])
        
    # coarse stage에서 특별하게 하는게 없어짐
    # 그래도 형식상 나누자
    if "coarse" in stage:   
        if is_training:
            means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor, feat, grid_offsets, grid_scaling, is_training=is_training, cam_type=cam_type)
        else:
            means3D_final, color, opacity_final, scales_final, rotations_final = \
                generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor, feat, grid_offsets, grid_scaling, is_training=is_training, cam_type=cam_type)
    
    elif "fine" in stage:
        # time 정보
        if cam_type != "PanopticSports":
            time = torch.tensor(viewpoint_camera.time).to(anchor.device).repeat(anchor.shape[0],1) # time을 anchor 길이 만큼만 repeat하면 됨
        else:
            time = torch.tensor(viewpoint_camera['time']).to(anchor.device).repeat(anchor.shape[0],1)

        ####################################################
        # 대망의 feature deformation        
        # v1 - deformation only using anchor point and Hexplane
        # anchor_deformed, feat_deformed = pc._deformation(anchor, feat, time)
        
        # v2 = deform grid offset and scaling 
        anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed = pc._deformation(anchor, feat, grid_offsets, grid_scaling, time)

        ####################################################
      
        # deform된 anchor와 feat를 가지고 neural gaussian을 생성
        if is_training:
            means3D_final, color, opacity_final, scales_final, rotations_final, neural_opacity, mask = \
                generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)
        else:
            means3D_final, color, opacity_final, scales_final, rotations_final = \
                generate_neural_gaussians_v2(viewpoint_camera, pc, visible_mask, anchor_deformed, feat_deformed, grid_offsets_deformed, grid_scaling_deformed, is_training=is_training, cam_type=cam_type)     

    
    # 이후 과정은 공통임
    # screenspace point 생성    
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
    
    if pipe.compute_cov3D_python: # False임
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    
    # 얘는 무조건 함
    rotations_final = pc.rotation_activation(rotations_final)

    # 1인경우 activation 하도록    
    if gvc_params["GVC_Opacity_Activation"] == 1:
        opacity_final = pc.opacity_activation(opacity_final)
        
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
        return rendered_image, screenspace_points, radii, mask, neural_opacity, scales_final, depth
    else:
        return rendered_image, screenspace_points, radii, depth



# GVC test mode 1일 때 호출되는 render 함수
def render_test1(gvc_params, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0,
              override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False):
    is_training = pc.get_color_mlp.training
           
    if gvc_params["GVC_Scale_Activation"] == 0:
    # anchor 인근의 neural gaussian을 생성한다. 
    # 기본 Scaffold-GS와 동일한 모드
        if is_training:
            xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians_v0(viewpoint_camera, pc, visible_mask, is_training=is_training)
        else:
            xyz, color, opacity, scaling, rot = generate_neural_gaussians_v0(viewpoint_camera, pc, visible_mask, is_training=is_training) 
    elif gvc_params["GVC_Scale_Activation"] == 1:
        # modified!!!!
        # Scale activation도 외부에서 수행
        if is_training:
            xyz, color, opacity, scaling_repeat, scaling, rot, neural_opacity, mask = generate_neural_gaussians_v1(viewpoint_camera, pc, visible_mask, is_training=is_training)
        else:
            xyz, color, opacity, scaling_repeat, scaling, rot = generate_neural_gaussians_v1(viewpoint_camera, pc, visible_mask, is_training=is_training) 
    else:
        raise NotImplementedError

    # screenspace point 생성
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # mean3D 부터 먼저 할당
    means3D = xyz

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
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera'] # PanopticSports인 경우에는 camera에 다 들어있나보다
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Scaffold-GS
    means2D = screenspace_points 
    # opacity = opacity # neural gaussian에서 생성된 opacity
    shs = None

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    if pipe.compute_cov3D_python: # False임
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        '''
        # original 4DGS
        scales = pc._scaling
        rotations = pc._rotation
        '''
        # Scaffold-GS
        scales = scaling
        rotations = rot

    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        # print("coarse stage: no deformation")
        # print("means3D_final, scales_final, rotations_final, opacity_final, shs_final",means3D_final.shape, scales_final.shape, rotations_final.shape, opacity_final.shape, shs_final.shape) 
        # torch.Size([37353, 3]) torch.Size([37353, 3]) torch.Size([37353, 4]) torch.Size([37353, 1]) torch.Size([37353, 16, 3])
        # Keyframe의 Gaussian attributes torch tensor들을 가지고 있다고 보면 됨
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                rotations, opacity, shs,
                                                                time)
        # time2 = get_time()
        # print("asset value:",time2-time1)
        # 각각의 attributes에 대한 사전 정의된 activation function을 적용
        # Scaffold-GS에는 activation function이 없으므로 fine stage에서만 적용

    else:
        raise NotImplementedError

    # 얘는 무조건 함
    rotations_final = pc.rotation_activation(rotations_final)

    # fine elif 문 안에 있던걸 밖으로 뺐음
    if gvc_params["GVC_Scale_Activation"] == 1:
        #scales_final = pc.scaling_activation(scales_final) 
        scales_final = scaling_repeat * torch.sigmoid(scales_final) 

    # 1인경우 activation 하도록    
    if gvc_params["GVC_Opacity_Activation"] == 1:
        opacity_final = pc.opacity_activation(opacity_final)

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
        return rendered_image, screenspace_points, radii, mask, neural_opacity, scaling, depth
    else:
        return rendered_image, screenspace_points, radii, depth
    

# original render function
def render_original(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           override_color = None, stage="fine", cam_type=None, visible_mask=None, retain_grad=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
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
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera'] # PanopticSports인 경우에는 camera에 다 들어있나보다
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # add deformation to each points
    # deformation = pc.get_deformation

    
    means2D = screenspace_points 
    opacity = pc._opacity
    shs = pc.get_features

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python: # False임
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
        # print("coarse stage: no deformation")
        # print("means3D_final, scales_final, rotations_final, opacity_final, shs_final",means3D_final.shape, scales_final.shape, rotations_final.shape, opacity_final.shape, shs_final.shape) 
        # torch.Size([37353, 3]) torch.Size([37353, 3]) torch.Size([37353, 4]) torch.Size([37353, 1]) torch.Size([37353, 16, 3])
        # Keyframe의 Gaussian attributes torch tensor들을 가지고 있다고 보면 됨
    elif "fine" in stage:
        # time0 = get_time()
        # means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point], 
        #                                                                  rotations[deformation_point], opacity[deformation_point],
        #                                                                  time[deformation_point])
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError



    # time2 = get_time()
    # print("asset value:",time2-time1)
    # 각각의 attributes에 대한 사전 정의된 activation function을 적용
    scales_final = pc.scaling_activation(scales_final) 
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    # print(opacity.max())
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
            # shs = 
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # time3 = get_time()
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    # time4 = get_time()
    # print("rasterization:",time4-time3)
    # breakpoint()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth}

