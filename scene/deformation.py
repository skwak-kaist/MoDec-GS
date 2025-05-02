import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.graphics_utils import apply_rotation, batch_quaternion_multiply
from scene.hexplane import HexPlaneField
from scene.grid import DenseGrid
#from scene.grid import HashHexPlane

class GaussianDeformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None, deform_stage="global"):
        super(GaussianDeformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.deform_stage = deform_stage
        if self.deform_stage == "global":
            self.grid = HexPlaneField(args.bounds, args.kplanes_config_global, args.multires_global)
        elif self.deform_stage == "local":
            self.grid = HexPlaneField(args.bounds, args.kplanes_config_local, args.multires_local)
        else: 
            assert False, "Invalid deform stage"
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("GaussianDeformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        self.opacity_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 

        hidden = self.feature_out(hidden)   
 
        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(opacity_emb[:,0]).unsqueeze(-1)
        # breakpoint()
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            dx = self.pos_deform(hidden)
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        if self.args.no_do :
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
          
            opacity = torch.zeros_like(opacity_emb[:,:1])
            opacity = opacity_emb[:,:1]*mask + do       
        shs = None

        return pts, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    
class deform_network_gaussian(nn.Module):
    def __init__(self, args, deform_stage="global"):
        super(deform_network_gaussian, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = GaussianDeformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args, deform_stage=deform_stage)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)
    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations, opacity, shs = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

# Anchor deformation class
class DeformationAnchor(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None, deform_stage="global"):
        super(DeformationAnchor, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.deform_stage = deform_stage
        if self.deform_stage == "global":
            self.grid = HexPlaneField(args.bounds, args.kplanes_config_global, args.multires_global)
        elif self.deform_stage == "local":
            self.grid = HexPlaneField(args.bounds, args.kplanes_config_local, args.multires_local)
        else: 
            assert False, "Invalid deform stage"
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        self.ratio=0
        self.create_net()
        
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("DeformationAnchor Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        
        if self.args.anchor_deform:        
            self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        if self.args.local_context_feature_deform:    
            self.feature_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, self.args.deform_feat_dim))
        if self.args.grid_offsets_deform:
            self.grid_offsets_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, self.args.deform_n_offsets*3))
        if self.args.grid_scale_deform:
            self.grid_scaling_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 6))
        
        
        if self.args.anchor_dynamics_position or self.args.anchor_dynamics_scaling or self.args.anchor_dynamics_offset or self.args.anchor_dynamics_local_context_feature: 
            if self.args.dynamics_activation == "relu":
                self.dynamics_activation = nn.ReLU()
            elif self.args.dynamics_activation == "tanh":
                self.dynamics_activation = nn.Tanh()
            elif self.args.dynamics_activation == "sigmoid":
                self.dynamics_activation = nn.Sigmoid()
            else:
                assert False, "Invalid dynamics activation function"
        


    def get_dynamics(self, dynamics):
        # get global dynamics
        if self.args.anchor_dynamics_global_type == "mask":
            dynamics_global = ((torch.sigmoid(dynamics[:,0]) > 0.01).float() - torch.sigmoid(dynamics[:,0])).detach() + torch.sigmoid(dynamics[:,0])
        elif self.args.anchor_dynamics_global_type == "mul":
            dynamics_global = self.dynamics_activation(dynamics[:,0])
        else: 
            assert False, "Invalid dynamics type"
            
        # get local dynamics
        if self.args.anchor_dynamics_local_type == "mask":
            dynamics_local = ((torch.sigmoid(dynamics[:,-1]) > 0.01).float() - torch.sigmoid(dynamics[:,-1])).detach() + torch.sigmoid(dynamics[:,-1])
        elif self.args.anchor_dynamics_local_type == "mul":
            dynamics_local = self.dynamics_activation(dynamics[:,-1])
        else:
            assert False, "Invalid dynamics type"
            
        return dynamics_global.unsqueeze(-1), dynamics_local.unsqueeze(-1)

    def query_time(self, rays_pts_emb, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
       
        hidden = self.feature_out(hidden)   

        return hidden
    @property
    def get_empty_ratio(self):
        return self.ratio
    
    def forward(self, rays_pts_emb, feat=None, grid_offsets=None, grid_scaling=None, dynamics=None, time_emb=None, time_feature=None): # revised
        if time_emb is None:
            breakpoint()
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, feat, grid_offsets, grid_scaling, dynamics, time_emb, time_feature) # revised

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    
    def forward_dynamic(self, rays_pts_emb, feat, grid_offsets, grid_scaling, dynamics, time_emb, time_feature): # revised
        # query grid feature by inputting ray point and time
        hidden = self.query_time(rays_pts_emb, time_emb) 
        
        # mask for empty voxel
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(rays_pts_emb[:,0]).unsqueeze(-1) # revised
        
        # dynamics mask
        if self.args.anchor_dynamics_position or self.args.anchor_dynamics_scaling or self.args.anchor_dynamics_offset or self.args.anchor_dynamics_local_context_feature:                 
            dynamics_anchor, dynamics_feature = self.get_dynamics(dynamics)
        
        # Deformation process HERE!
        if self.args.anchor_deform:
            dx = self.pos_deform(hidden)
            
            if self.args.anchor_dynamics_position:
                dx = dx * dynamics_anchor
            
            pts = torch.zeros_like(rays_pts_emb[:,:3])
            pts = rays_pts_emb[:,:3]*mask + dx
        else:
            pts = rays_pts_emb[:,:3]
            
        if self.args.local_context_feature_deform:    
            df = self.feature_deform(hidden)
            
            if self.args.anchor_dynamics_local_context_feature:
                df = df * dynamics_feature
            
            feat_deformed = torch.zeros_like(feat)
            feat_deformed = feat*mask + df
        else:
            feat_deformed = feat
            
        if self.args.grid_offsets_deform:
            do = self.grid_offsets_deform(hidden)
            
            if self.args.anchor_dynamics_offset:
                do = do * dynamics_feature
            
            # Shape of grid offset [batch, n_offsets, 3] --> [batch, n_offsets*3]           
            grid_offsets = grid_offsets.reshape([-1, self.args.deform_n_offsets*3])
            grid_offsets_deformed = torch.zeros_like(grid_offsets)
            grid_offsets_deformed = grid_offsets * mask + do
            
            # Reshape back to [batch, n_offsets, 3]
            grid_offsets_deformed = grid_offsets_deformed.reshape([-1, self.args.deform_n_offsets, 3])
        else:
            grid_offsets_deformed = grid_offsets
            
        if self.args.grid_scale_deform:
            ds = self.grid_scaling_deform(hidden)
            
            if self.args.anchor_dynamics_scaling:
                ds = ds * dynamics_feature
            
            grid_scaling_deformed = torch.zeros_like(grid_scaling)
            grid_scaling_deformed = grid_scaling*mask + ds
        else:
            grid_scaling_deformed = grid_scaling
        
        return pts, feat_deformed, grid_offsets_deformed, grid_scaling_deformed
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list


class deform_network_anchor(nn.Module):
    def __init__(self, args, deform_stage="global"): 
        super(deform_network_anchor, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        
        posbase_pe= args.posebase_pe

        # scaffold-gs
        
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        
        # Be careful of the input/output of the Deformation field!
        self.deformation_net = DeformationAnchor(W=net_width, 
                                                    D=defor_depth, 
                                                    input_ch=(3)+(3*(posbase_pe))*2, 
                                                    grid_pe=grid_pe, 
                                                    input_ch_time=timenet_output, 
                                                    args=args,
                                                    deform_stage=deform_stage)
        
        
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.apply(initialize_weights)

    def forward(self, anchor, feat=None, grid_offsets=None, grid_scaling=None, dynamics=None, times_sel=None):
        return self.forward_dynamic(anchor, feat, grid_offsets, grid_scaling, dynamics, times_sel)
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    
       
    def forward_dynamic(self, anchor, feat=None, grid_offsets=None, grid_scaling=None, dynamics=None, times_sel=None):       
        point_emb = poc_fre(anchor, self.pos_poc)
        time_emb = poc_fre(times_sel, self.time_poc) 
        times_feature = self.timenet(time_emb) # why isn't this used?
        '''
        means3D, scales, rotations, opacity, shs = self.deformation_net(point_emb,
                                                scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                None,
                                                times_sel)
        '''
        # here!!
        anchor, feat, grid_offsets, grid_scaling = self.deformation_net(point_emb, 
                                                                        feat, 
                                                                        grid_offsets, 
                                                                        grid_scaling, 
                                                                        dynamics, 
                                                                        times_sel)
        #anchor, feat = self.deformation_net(point_emb, feat, grid_offsets, grid_scaling, time_emb) # time positional encoding
        return anchor, feat, grid_offsets, grid_scaling
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb