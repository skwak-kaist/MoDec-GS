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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_canonical
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        
        rendering = render(view, gaussians, pipeline, background, cam_type=cam_type)["render"]

        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)
    
    scene_name = os.path.basename(model_path)
    test_case_name = os.path.basename(os.path.split(model_path)[0])

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), test_case_name + '_' + scene_name + '_' + name + '_' + 'video_rgb.mp4'), render_images, fps=30)


def render_set_canonical(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    ### local canonical_rendering_still ###
    render_path = os.path.join(model_path, name, "ours_{}_canonical".format(iteration), "renders_local_canonical_still")
    
    makedirs(render_path, exist_ok=True)
    
    render_images = []
    render_list = []
    
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        if idx == 0:time1 = time()
        if idx == 0:
            initla_camera = view
        
        rendering = render_canonical(view, gaussians, pipeline, background, cam_type=cam_type, canonical_times='local_still', still_camera=initla_camera)["render"]

        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    #multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)
    
    scene_name = os.path.basename(model_path)
    test_case_name = os.path.basename(os.path.split(model_path)[0])

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}_canonical".format(iteration), test_case_name + '_' + scene_name + '_' + name + '_' + 'local_canonical_still_video_rgb.mp4'), render_images, fps=30)   
        
    ### global canonical_rendering ###
    render_path = os.path.join(model_path, name, "ours_{}_canonical".format(iteration), "renders_global_canonical")
    makedirs(render_path, exist_ok=True)
    
    render_images = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        
        rendering = render_canonical(view, gaussians, pipeline, background, cam_type=cam_type, canonical_times='global')["render"]

        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    #multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)
    
    scene_name = os.path.basename(model_path)
    test_case_name = os.path.basename(os.path.split(model_path)[0])

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}_canonical".format(iteration), test_case_name + '_' + scene_name + '_' + name + '_' + 'global_canonical_video_rgb.mp4'), render_images, fps=30)   
    
    ### local canonical_rendering ###
    render_path = os.path.join(model_path, name, "ours_{}_canonical".format(iteration), "renders_local_canonical")
    makedirs(render_path, exist_ok=True)
    
    render_images = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        rendering = render_canonical(view, gaussians, pipeline, background, cam_type=cam_type, canonical_times='local')["render"]

        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    #multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)
    
    scene_name = os.path.basename(model_path)
    test_case_name = os.path.basename(os.path.split(model_path)[0])

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}_canonical".format(iteration), test_case_name + '_' + scene_name + '_' + name + '_' + 'local_canonical_video_rgb.mp4'), render_images, fps=30)
    
    ### rendering_still ###
    render_path = os.path.join(model_path, name, "ours_{}_canonical".format(iteration), "renders_still")
    makedirs(render_path, exist_ok=True)
    
    render_images = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        rendering = render_canonical(view, gaussians, pipeline, background, cam_type=cam_type, canonical_times='render_still', still_camera=initla_camera)["render"]
        
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    #multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)
    
    scene_name = os.path.basename(model_path)
    test_case_name = os.path.basename(os.path.split(model_path)[0])

    imageio.mimwrite(os.path.join(model_path, name, "ours_{}_canonical".format(iteration), test_case_name + '_' + scene_name + '_' + name + '_' + 'render_still_video_rgb.mp4'), render_images, fps=30)
    

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, canonical_frame_render : bool):
    with torch.no_grad():
        # Create Gaussian model
        gaussians = GaussianModel(hyperparam, dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, 
                                  dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, 
                                  dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, 
                                  dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
                                  dataset.anchor_dynamics, dataset.anchor_dynamics_share, 
                                  dataset.GLMD, dataset.GAD_method, dataset.LGD_method,
                                  dataset.TIA, dataset.TIA_num_segments, dataset.TIA_step_size, dataset.TIA_threshold)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)

        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)

        if canonical_frame_render:
            #render_set_canonical(dataset.model_path, "video",scene.loaded_iter, scene.getVideoCameras(),gaussians, pipeline, background, cam_type)
            render_set_canonical(dataset.model_path, "test",scene.loaded_iter, scene.getTestCameras(),gaussians, pipeline, background, cam_type)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--canonical_frame_render", action="store_true")
    parser.add_argument("--configs", type=str)

    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
      

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.canonical_frame_render)
