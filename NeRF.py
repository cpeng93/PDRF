import torch
import torch.nn as nn
from run_nerf_helpers import *
import os
import imageio
import time, pickle
from torch_efficient_distloss import flatten_eff_distloss

import random
import numpy as np
seed = 10000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.default_rng(seed=seed)

def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class BlurModel(nn.Module):
    def __init__(self, num_img, poses, num_pt, kernel_hwindow, kernel_type, img_wh=None, random_hwindow=0.25,
                 in_embed=3, random_mode='input', img_embed=32, spatial_embed=0, depth_embed=0,
                 num_hidden=3, num_wide=64, feat_cnl=15, short_cut=False, pattern_init_radius=0.1,
                 isglobal=False, optim_trans=False, optim_spatialvariant_trans=False):
        """
        num_img: number of image, used for deciding the view embedding
        poses: the original poses, used for generating new rays, len(poses) == num_img
        num_pt: number of sparse point, we use 5 in the paper
        kernel_hwindow: the size of physically equivalent blur kernel, the sparse points are bounded inside the blur kernel. 
                        Can be a very big number
        
        random_hwindow: in training, we randomly perturb the sparse point to model a smooth manifold
        random_mode: 'input' or 'output', it controls whether the random perturb is added to the input of DSK or output of DSK
        // the above two parameters do not have big impact on the results

        in_embed: embedding for the canonical kernel location
        img_embed: the length of the view embedding
        spatial_embed: embedding for the pixel location of the blur kernel inside an image
        depth_embed: (deprecated) the embedding for the depth of current rays
        
        num_hidden, num_wide, short_cut: control the structure of the MLP
        pattern_init_radius: the little gain add to the deform location described in Sec. 4.4
        isglobal: control whether the canonical kernel should be shared by all the input views or not, does not have big impact on the results
        optim_trans: whether to optimize the ray origin described in Sec. 4.3
        optim_spatialvariant_trans: whether to optimize the ray origin for each view or each kernel point. 
        """
        super().__init__()
        self.num_pt = num_pt
        self.num_img = num_img
        self.short_cut = short_cut
        self.kernel_hwindow = kernel_hwindow
        self.random_hwindow = random_hwindow  # about 1 pix
        self.random_mode = random_mode
        self.kernel_type = kernel_type
        self.isglobal = isglobal
        self.feat_cnl = feat_cnl
        pattern_num = 1 if isglobal else num_img
        assert self.random_mode in ['input', 'output'], f"BlurModel::random_mode {self.random_mode} unrecognized, " \
                                                        f"should be input/output"
        self.register_buffer("poses", poses)
        self.register_parameter("pattern_pos",
                                nn.Parameter(torch.randn(pattern_num, num_pt, 2)
                                             .type(torch.float32) * pattern_init_radius, True))
        self.optim_trans = optim_trans
        self.optim_sv_trans = optim_spatialvariant_trans

        if optim_trans:
            self.register_parameter("pattern_trans",
                                    nn.Parameter(torch.zeros(pattern_num, num_pt, 2)
                                                 .type(torch.float32), True))

        if in_embed > 0:
            self.in_embed_fn, self.in_embed_cnl = get_embedder(in_embed, input_dim=2)
        else:
            self.in_embed_fn, self.in_embed_cnl = None, 0

        self.img_embed_cnl = img_embed

        if spatial_embed > 0:
            self.spatial_embed_fn, self.spatial_embed_cnl = get_embedder(spatial_embed, input_dim=2)
        else:
            self.spatial_embed_fn, self.spatial_embed_cnl = None, 0

        if depth_embed > 0:
            self.require_depth = True
            self.depth_embed_fn, self.depth_embed_cnl = get_embedder(depth_embed, input_dim=1)
        else:
            self.require_depth = False
            self.depth_embed_fn, self.depth_embed_cnl = None, 0

        in_cnl = self.in_embed_cnl + self.img_embed_cnl + self.depth_embed_cnl + self.spatial_embed_cnl
        if self.kernel_type == 'PBE':
            in_cnl += self.feat_cnl
        out_cnl = 1 + 2 + 2 if self.optim_sv_trans else 1 + 2  # u, v, w or u, v, w, dx, dy
        hiddens = [nn.Linear(num_wide, num_wide) if i % 2 == 0 else nn.ReLU()
                   for i in range((num_hidden - 1) * 2)]
        self.linears = nn.Sequential(
            nn.Linear(in_cnl, num_wide), nn.ReLU(),
            *hiddens,
        )
        self.linears1 = nn.Sequential(
            nn.Linear((num_wide + in_cnl) if short_cut else num_wide, num_wide), nn.ReLU(),
            nn.Linear(num_wide, out_cnl)
        )
        self.linears.apply(init_linear_weights)
        self.linears1.apply(init_linear_weights)
        if img_embed > 0:
            self.register_parameter("img_embed",
                                    nn.Parameter(torch.zeros(num_img, img_embed).type(torch.float32), True))
        else:
            self.img_embed = None

    def forward(self, H, W, K, rays, rays_info,feats=None):
        """
        inputs: all input has shape (ray_num, cnl)
        outputs: output shape (ray_num, ptnum, 3, 2)  last two dim: [ray_o, ray_d]
        """
        img_idx = rays_info['images_idx'].squeeze(-1)
        img_embed = self.img_embed[img_idx] if self.img_embed is not None else \
            torch.tensor([]).reshape(len(img_idx), self.img_embed_cnl)

        pt_pos = self.pattern_pos.expand(len(img_idx), -1, -1) if self.isglobal \
            else self.pattern_pos[img_idx]
        pt_pos = torch.tanh(pt_pos) * self.kernel_hwindow

        if self.random_hwindow > 0 and self.random_mode == "input":
            random_pos = torch.randn_like(pt_pos) * self.random_hwindow
            pt_pos = pt_pos + random_pos

        input_pos = pt_pos  # the first point is the reference point
        if self.in_embed_fn is not None:
            pt_pos = pt_pos * (np.pi / self.kernel_hwindow)
            pt_pos = self.in_embed_fn(pt_pos)

        img_embed_expand = img_embed[:, None].expand(len(img_embed), self.num_pt, self.img_embed_cnl)

        if self.kernel_type == 'DSK':
            x = torch.cat([pt_pos, img_embed_expand], dim=-1)
        else:
            if feats == None:
                x = torch.cat([pt_pos, img_embed_expand,torch.zeros(len(img_embed), self.num_pt, self.feat_cnl)], dim=-1)
            else:
                x = torch.cat([pt_pos, img_embed_expand,feats.view(len(img_embed), self.num_pt,-1)], dim=-1)

        rays_x, rays_y = rays_info['rays_x'], rays_info['rays_y']
        if self.spatial_embed_fn is not None:
            spatialx = rays_x / (W / 2 / np.pi) - np.pi
            spatialy = rays_y / (H / 2 / np.pi) - np.pi  # scale 2pi to match the freq in the embedder
            spatial_save = torch.cat([spatialx, spatialy], dim=-1)
            spatial = self.spatial_embed_fn(spatial_save)
            spatial = spatial[:, None].expand(len(img_idx), self.num_pt, self.spatial_embed_cnl)
            x = torch.cat([x, spatial], dim=-1)

        if self.depth_embed_fn is not None:
            depth = rays_info['ray_depth']
            depth = depth * np.pi
            depth = self.depth_embed_fn(depth)
            depth = depth[:, None].expand(len(img_idx), self.num_pt, self.depth_embed_cnl)
            x = torch.cat([x, depth], dim=-1)

        # forward
        x1 = self.linears(x)
        x1 = torch.cat([x, x1], dim=-1) if self.short_cut else x1
        x1 = self.linears1(x1)

        delta_trans = None
        if self.optim_sv_trans:
            delta_trans, delta_pos, weight = torch.split(x1, [2, 2, 1], dim=-1)
        else:
            delta_pos, weight = torch.split(x1, [2, 1], dim=-1)

        if self.optim_trans:
            delta_trans = self.pattern_trans.expand(len(img_idx), -1, -1) if self.isglobal \
                else self.pattern_trans[img_idx]

        if delta_trans is None:
            delta_trans = torch.zeros_like(delta_pos)

        delta_trans = delta_trans * 0.01
        new_rays_xy = delta_pos + input_pos
        if self.kernel_type == 'PBE':
            new_rays_xy[:, 0, :] = 0
            delta_trans[:, 0, :] = 0
            align = None
        else:
            align = new_rays_xy[:, 0, :].abs().mean()
            align += (delta_trans[:, 0, :].abs().mean() * 10)
        weight = torch.softmax(weight[..., 0], dim=-1)

        if self.random_hwindow > 0 and self.random_mode == 'output':
            raise NotImplementedError(f"{self.random_mode} for self.random_mode is not implemented")

        poses = self.poses[img_idx]
        # get rays from offsetted pt position
        rays_x = (rays_x - K[0, 2] + new_rays_xy[..., 0]) / K[0, 0]
        rays_y = -(rays_y - K[1, 2] + new_rays_xy[..., 1]) / K[1, 1]
        dirs = torch.stack([rays_x - delta_trans[..., 0],
                            rays_y - delta_trans[..., 1],
                            -torch.ones_like(rays_x)], -1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * poses[..., None, :3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        translation = torch.stack([
            delta_trans[..., 0],
            delta_trans[..., 1],
            torch.zeros_like(rays_x),
            torch.ones_like(rays_x)
        ], dim=-1)
        rays_o = torch.sum(translation[..., None, :] * poses[:, None], dim=-1)

        return torch.stack([rays_o, rays_d], dim=-1), weight, align


class NeRFAll(nn.Module):
    def __init__(self, args, kernelsnet=None):
        super().__init__()
        self.args = args
        self.embed_fn, self.input_ch = get_embedder(args.multires, args.i_embed)

        self.kernel_type = args.kernel_type
        self.mode = args.mode
        self.input_ch_views = 0
        self.kernelsnet = kernelsnet
        self.embeddirs_fn = None
        if args.use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(args.multires_views, args.i_embed)

        self.output_ch = 5 if args.N_importance > 0 else 4

        skips = [4]
        if self.mode == 'c2f':
            self.mlp_coarse = NeRFSmall_ray(aabb=args.bounding_box,
                            num_layers=args.coarse_num_layers,
                            hidden_dim=args.coarse_hidden_dim,
                            geo_feat_dim=args.kernel_feat_cnl,
                            num_layers_color=args.coarse_num_layers_color,
                            hidden_dim_color=args.coarse_hidden_dim_color,
                            input_ch=args.coarse_app_dim+self.input_ch, input_ch_views=self.input_ch_views,
                            render_rmnearplane=args.render_rmnearplane,app_dim=args.coarse_app_dim,
                            app_n_comp=args.coarse_app_n_comp, n_voxels=args.coarse_n_voxels)


            grad_vars_vol, grad_vars_net = self.mlp_coarse.get_optparam_groups()
            if self.kernelsnet != None:
                self.grad_vars = grad_vars_net + list(self.kernelsnet.parameters())
            else:
                self.grad_vars = grad_vars_net
            self.grad_vars_vol = grad_vars_vol


            self.mlp_fine = NeRFSmall_voxel(aabb=args.bounding_box,
                        num_layers=args.fine_num_layers,
                        hidden_dim=args.fine_hidden_dim,
                        geo_feat_dim=args.fine_geo_feat_dim,
                        num_layers_color=args.fine_num_layers_color,
                        hidden_dim_color=args.fine_hidden_dim_color,
                        input_ch=args.coarse_app_dim+args.fine_app_dim+self.input_ch, input_ch_views=self.input_ch_views,
                        render_rmnearplane=args.render_rmnearplane,app_dim=args.fine_app_dim,
                        app_n_comp=args.fine_app_n_comp, n_voxels=args.fine_n_voxels)

            
            grad_vars_vol, grad_vars_net = self.mlp_fine.get_optparam_groups()
            self.grad_vars += grad_vars_net
            self.grad_vars_vol += grad_vars_vol
        elif self.mode == 'nerf':
            self.mlp_coarse = NeRF(
                D=args.netdepth, W=args.netwidth,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs,
                rgb_activate=args.rgb_activate,sigma_activate=args.sigma_activate,
                render_rmnearplane=args.render_rmnearplane)

            self.mlp_fine = NeRF(
                D=args.netdepth_fine, W=args.netwidth_fine,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs,
                rgb_activate=args.rgb_activate,sigma_activate=args.sigma_activate,
                render_rmnearplane=args.render_rmnearplane)
        else:
            raise NotImplementedError(f"{self.mode} for rendering network is not implemented")



        activate = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'exp': torch.exp, 'none': lambda x: x,
                    'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
                    'softplus': lambda x: nn.Softplus()(x - 1)}
        self.rgb_activate = activate[args.rgb_activate]
        self.sigma_activate = activate[args.sigma_activate]
        self.tonemapping = ToneMapping(args.tone_mapping_type)
        print(self.mlp_coarse,self.mlp_fine,self.kernelsnet)


    def render_rays(self,
                    ray_batch,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).type_as(rays_o)

            z_vals = lower + (upper - lower) * t_rand

        pts0 = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        if self.mode == 'c2f':
            ft_coarse = self.mlp_coarse.sample(pts0)
            ft_fine = self.mlp_fine.sample(pts0)
            ft_comb0 = torch.cat([ft_coarse,ft_fine],-1)

            rgb_map_0, depth_map_0, acc_map_0, weights_coarse, _ = self.mlp_coarse(pts0, viewdirs, ft_coarse, self.embed_fn, self.embeddirs_fn, z_vals, rays_d, raw_noise_std, self.training)


            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, order = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts1 = rays_o[..., None, :] + rays_d[..., None, :] * z_samples[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

            ft_coarse = self.mlp_coarse.sample(pts1)
            pts = torch.cat([pts0, pts1],1)[torch.arange(pts1.shape[0]).unsqueeze(1),order]
            ft_fine = self.mlp_fine.sample(pts1)
            ft_comb1 = torch.cat([ft_coarse,ft_fine],-1)
            ft_comb = torch.cat([ft_comb0, ft_comb1],1)[torch.arange(pts1.shape[0]).unsqueeze(1),order]

            rgb_map, depth_map, acc_map, weights_fine = self.mlp_fine(pts, viewdirs, ft_comb, self.embed_fn, self.embeddirs_fn, z_vals, rays_d, raw_noise_std, self.training)
        else:
            rgb_map, depth_map, acc_map, weights, _ = self.mlp_coarse(pts0, viewdirs, self.embed_fn,self.embeddirs_fn, z_vals, rays_d, raw_noise_std, white_bkgd, self.training)

            rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

            # mlp = self.mlp_coarse if self.mlp_fine is None else self.mlp_fine
            # raw = self.mlpforward(pts, viewdirs, mlp)

            rgb_map, depth_map, acc_map, weights, _ = self.mlp_fine(pts, viewdirs, self.embed_fn, self.embeddirs_fn, z_vals, rays_d, raw_noise_std, white_bkgd, self.training)


        ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map}
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def forward(self, H, W, K, chunk=1024 * 32, rays=None, rays_info=None, poses=None, **kwargs):
        """
        render rays or render poses, rays and poses should atleast specify one
        calling model.train() to render rays, where rays, rays_info, should be specified
        calling model.eval() to render an image, where poses should be specified

        optional args:
        force_naive: when True, will only run the naive NeRF, even if the kernelsnet is specified

        """
        # training
        if self.training:
            assert rays is not None, "Please specify rays when in the training mode"

            force_baseline = kwargs.pop("force_naive", True)
            # kernel mode, run multiple rays to get result of one ray
            if self.kernelsnet is not None and not force_baseline:
                if self.kernel_type == 'PBE':
                    new_rays0, weight0, _ = self.kernelsnet(H, W, K, rays, rays_info) 
                    ray_num, pt_num = new_rays0.shape[:2]

                    rgb0, features = self.coarse_render(H, W, K, chunk, new_rays0.reshape(-1, 3, 2), **kwargs)

                    rgb0_pts = rgb0.reshape(ray_num, pt_num, 3)
                    rgb0 = torch.sum(rgb0_pts * weight0[..., None], dim=1)
                    rgb0 = self.tonemapping(rgb0)
                else:
                    features = None

                new_rays, weight1, align_loss = self.kernelsnet(H, W, K, rays, rays_info,feats=features) 
                ray_num, pt_num = new_rays.shape[:2]
                rgb, depth, acc, extras = self.render(H, W, K, chunk, new_rays.reshape(-1, 3, 2), **kwargs)
                rgb_pts = rgb.reshape(ray_num, pt_num, 3)
                rgb1_pts = extras['rgb0'].reshape(ray_num, pt_num, 3)
                rgb = torch.sum(rgb_pts * weight1[..., None], dim=1)
                rgb1 = torch.sum(rgb1_pts * weight1[..., None], dim=1)
                rgb = self.tonemapping(rgb)
                rgb1 = self.tonemapping(rgb1)
                if self.kernel_type == 'PBE':
                    rgb1 = (rgb0 + rgb1)/2

                other_loss = {}
                if self.mode == 'c2f':
                    other_loss["TV"] = (self.mlp_fine.TV_loss_app()+self.mlp_coarse.TV_loss_app()) * 5
                if align_loss is not None:
                    other_loss["align"] = align_loss.reshape(1, 1)                
                return rgb, rgb1, other_loss#, [weight]+trace
            else:
                rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                other_loss = {}
                if self.mode == 'c2f':
                    other_loss["TV"] = (self.mlp_fine.TV_loss_app()+self.mlp_coarse.TV_loss_app()) * 5
                return self.tonemapping(rgb), self.tonemapping(extras['rgb0']), other_loss

        #  evaluation
        else:
            assert poses is not None, "Please specify poses when in the eval model"
            rgbs, depths = self.render_path(H, W, K, chunk, poses, **kwargs)
            return self.tonemapping(rgbs), depths

    def render(self, H, W, K, chunk, rays=None, c2w=None, ndc=True,
               near=0., far=1.,
               use_viewdirs=False, c2w_staticcam=None,
               **kwargs):  # the render function
        """Render rays
            Args:
              H: int. Height of image in pixels.
              W: int. Width of image in pixels.
              focal: float. Focal length of pinhole camera.
              chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
              rays: array of shape [2, batch_size, 3]. Ray origin and direction for
                each example in batch.
              c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
              ndc: bool. If True, represent ray origin, direction in NDC coordinates.
              near: float or array of shape [batch_size]. Nearest distance for a ray.
              far: float or array of shape [batch_size]. Farthest distance for a ray.
              use_viewdirs: bool. If True, use viewing direction of a point in space in model.
              c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
               camera while using other c2w argument for viewing directions.
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays[..., 0], rays[..., 1]

        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Batchfy and Render and reshape
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            # print(k,k_sh,all_ret[k].shape)
            if all_ret[k].shape[0] == 5120:
                continue
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        # ret_dict['dist'] = dist_loss
        return ret_list + [ret_dict]



    def coarse_render(self, H, W, K, chunk, rays=None, c2w=None, ndc=True,
               near=0., far=1.,
               use_viewdirs=False, c2w_staticcam=None,
               **kwargs):  # the render function
        """Render rays
            Args:
              H: int. Height of image in pixels.
              W: int. Width of image in pixels.
              focal: float. Focal length of pinhole camera.
              chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
              rays: array of shape [2, batch_size, 3]. Ray origin and direction for
                each example in batch.
              c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
              ndc: bool. If True, represent ray origin, direction in NDC coordinates.
              near: float or array of shape [batch_size]. Nearest distance for a ray.
              far: float or array of shape [batch_size]. Farthest distance for a ray.
              use_viewdirs: bool. If True, use viewing direction of a point in space in model.
              c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
               camera while using other c2w argument for viewing directions.
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays[..., 0], rays[..., 1]

        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Batchfy and Render and reshape
        all_ret = {}
        rgb,feat = self.coarse_render_rays(rays, **kwargs)
        return rgb,feat



    def coarse_render_rays(self,
                    ray_batch,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).type_as(rays_o)

            # Pytest, overwrite u with numpy's fixed random numbers
            # pytest=True
            # if pytest:
            #     np.random.seed(0)
            #     t_rand = np.random.rand(*list(z_vals.shape))
            #     t_rand = torch.tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts0 = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        if self.mode == 'c2f':
            ft_coarse = self.mlp_coarse.sample(pts0)
            rgb_map, _, _, _, feat = self.mlp_coarse(pts0, viewdirs, ft_coarse, self.embed_fn, self.embeddirs_fn, z_vals, rays_d, raw_noise_std, self.training)
        else:
            rgb_map, _, _, _, feat = self.mlp_coarse(pts0, viewdirs, self.embed_fn,self.embeddirs_fn, z_vals, rays_d, raw_noise_std, white_bkgd, self.training)
        return rgb_map, feat

    def render_path(self, H, W, K, chunk, render_poses, render_kwargs, render_factor=0, ):
        """
        render image specified by the render_poses
        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []

        t = time.time()
        for i, c2w in enumerate(render_poses):
            print(i, time.time() - t)
            t = time.time()
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1)
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=rays, c2w=c2w[:3, :4], **render_kwargs)

            rgbs.append(rgb)
            depths.append(depth)
            # rgbs.append(extras['rgb0'])
            # depths.append(extras['depth0'])            
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths
