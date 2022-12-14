import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

HALF_PIX = 0.5
# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

class ToneMapping(nn.Module):
    def __init__(self, map_type: str):
        super(ToneMapping, self).__init__()
        assert map_type in ['none', 'gamma', 'learn', 'ycbcr']
        self.map_type = map_type
        if map_type == 'learn':
            self.linear = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 1)
            )

    def forward(self, x):
        if self.map_type == 'none':
            return x
        elif self.map_type == 'learn':
            ori_shape = x.shape
            x_in = x.reshape(-1, 1)
            res_x = self.linear(x_in) * 0.1
            x_out = torch.sigmoid(res_x + x_in)
            return x_out.reshape(ori_shape)
        elif self.map_type == 'gamma':
            return x ** (1. / 2.2)
        else:
            assert RuntimeError("map_type not recognized")

# Positional encoding (section 5.1)
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs):
        # print(f"input device: {inputs.device}, freq_bands device: {self.freq_bands.device}")
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))
        return torch.cat(outputs, -1)



def get_embedder(multires, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class TVLoss(torch.nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        count_w = max(count_w, 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,rgb_activate='sigmoid',sigma_activate='relu',render_rmnearplane = 0):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.render_rmnearplane = render_rmnearplane
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        activate = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'exp': torch.exp, 'none': lambda x: x,
                    'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
                    'softplus': lambda x: nn.Softplus()(x - 1)}
        self.rgb_activate = activate[rgb_activate]
        self.sigma_activate = activate[sigma_activate]


        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def mlpforward(self, inputs, viewdirs, embed_fn, embeddirs_fn, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
            """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = embed_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # batchify execution
        if netchunk is None:
            outputs_flat, feature_flat = self.eval(embedded)
        else:
            outputs_flat, feature_flat = [], []
            for i in range(0, embedded.shape[0], netchunk):
                output,feature = self.eval(embedded[i:i + netchunk])
                outputs_flat.append(output)
                feature_flat.append(feature)

            outputs_flat, feature_flat = torch.cat(outputs_flat, 0), torch.cat(feature_flat, 0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        feature = torch.reshape(feature_flat, list(inputs.shape[:-1]) + [feature_flat.shape[-1]])
        return outputs, feature

    def raw2outputs(self, raw, feature, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        def raw2alpha(raw_, dists_, act_fn):
            alpha_ = - torch.exp(-act_fn(raw_) * dists_) + 1.
            return torch.cat([alpha_, torch.ones_like(alpha_[:, 0:1])], dim=-1)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = self.rgb_activate(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., :-1, 3]) * raw_noise_std
            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.tensor(noise)

        density = self.sigma_activate(raw[..., :-1, 3] + noise)
        if not self.training and self.render_rmnearplane > 0:
            mask = z_vals[:, 1:]
            mask = mask > self.render_rmnearplane / 128
            mask = mask.type_as(density)
            density = mask * density

        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)
        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        feature_map = torch.sum(weights[..., None] * feature, -2)  # [N_rays, 3]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, feature_map, density, acc_map, weights, depth_map

    def eval(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs, feature

    def forward(self, pts, viewdirs, pts_embed, dirs_embed, z_vals, rays_d, raw_noise_std, white_bkgd, is_train):

        raw, feature = self.mlpforward(pts, viewdirs, pts_embed, dirs_embed)
        rgb_map, feature_map, density_map, acc_map, weights, depth_map = self.raw2outputs(raw, feature, z_vals, rays_d, raw_noise_std, white_bkgd)
           
        return rgb_map, depth_map, acc_map, weights, feature_map


class NeRFSmall_ray(nn.Module):
    def __init__(self,
                 aabb,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 render_rmnearplane=0,app_dim=32,
                 app_n_comp=[64,16,16], n_voxels=16777248):
        super(NeRFSmall_ray, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.render_rmnearplane = render_rmnearplane
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
        # self.aabb = torch.FloatTensor([[-2.0815, -2.3389, -1.0001], [2.2236,  2.0548,  1.0001]]).cuda()
        self.app_dim = app_dim #app_dim
        self.app_n_comp = app_n_comp#[48,12,12]


        self.aabb = torch.stack(aabb).cuda()
        xyz_min, xyz_max = aabb
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
        gridSize = ((xyz_max - xyz_min) / voxel_size).long().tolist()            
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize)#.to(self.device)
        print('Coarse Ray GridSize', self.gridSize)        

        # self.gridSize = [164*2, 167*2, 76*2]
        # self.aabbSize = self.aabb[1] - self.aabb[0]
        # self.invaabbSize = 2.0/self.aabbSize

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]
        self.reg = TVLoss()


        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False)#.to(device)

    def init_one_svd(self, n_component, gridSize, scale):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        # return plane_coef, line_coef
        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)


    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars_vol = list(self.app_line)+ list(self.app_plane)
        grad_vars_net = list(self.basis_mat.parameters())+list(self.color_net.parameters())+list(self.sigma_net.parameters())
        return grad_vars_vol, grad_vars_net


        
    def TV_loss_app(self):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + self.reg(self.app_plane[idx]) * 1e-2 + self.reg(self.app_line[idx]) * 1e-3
        return total



    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)


    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, is_train=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            feature_map: [num_rays, 3]. Estimated feature sum of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        def raw2alpha(raw_, dists_, act_fn):
            alpha_ = - torch.exp(-act_fn(raw_) * dists_) + 1.
            return torch.cat([alpha_, torch.ones_like(alpha_[:, 0:1])], dim=-1)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]
        # dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        feature = torch.relu(raw[..., 1:])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., :-1, 0]) * raw_noise_std


        density = torch.relu(raw[..., :-1, 0] + noise)
        # print(density.shape, raw.shape)
        if not is_train and self.render_rmnearplane > 0:
            mask = z_vals[:, 1:]
            mask = mask > self.render_rmnearplane / 128
            mask = mask.type_as(density)
            density = mask * density

        # print(density.shape, dists.shape)
        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)

        # alpha = raw2alpha(raw[..., :-1, 3] + noise, dists, act_fn=self.sigma_activate)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        feature_map = torch.sum(weights[..., None] * feature, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)

        return feature_map, density, acc_map, weights, depth_map#, sparsity_loss


    def sample(self, pts):
        xyz_sampled = (pts.reshape(-1,3)-self.aabb[0]) * self.invaabbSize - 1
        return self.compute_appfeature(xyz_sampled).reshape(pts.shape[0],pts.shape[1],-1)


    def forward(self, pts, viewdirs, fts, pts_embed, dirs_embed, z_vals, rays_d, raw_noise_std, is_train):


        # time1 = time.time()
        input_locs = torch.reshape(pts, [-1, pts.shape[-1]])
        input_locs = pts_embed(input_locs)

        # xyz_sampled = (pts.reshape(-1,3)-self.aabb[0]) * self.invaabbSize - 1
        # input_pts = self.compute_appfeature(xyz_sampled)
        # input_dirs = viewdirs[:, None].expand(pts.shape)
        input_dirs = torch.reshape(viewdirs, [-1, viewdirs.shape[-1]])
        input_dirs = dirs_embed(input_dirs)

        # time2 = time.time()

        # sigma
        h = torch.cat([fts.view(pts.shape[0]*pts.shape[1],-1),input_locs],-1)
        # h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        h = h.reshape(pts.shape[0], pts.shape[1], -1)

        # time3 = time.time()
        feature_map, density_map, acc_map, weights, depth_map = self.raw2outputs(h, z_vals, rays_d, raw_noise_std, is_train=is_train)

        # time4 = time.time()
        # color
        h = torch.cat([feature_map,input_dirs],-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
            
        color = torch.sigmoid(h)
        return color, depth_map, acc_map, weights,feature_map


class NeRFSmall_voxel(nn.Module):
    def __init__(self,
                 aabb,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 render_rmnearplane=0,app_dim=32,
                 app_n_comp=[64,16,16], n_voxels=134217984):
        super(NeRFSmall_voxel, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.render_rmnearplane = render_rmnearplane
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 self.geo_feat_dim features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
        self.aabb = torch.stack(aabb).cuda()
        xyz_min, xyz_max = aabb
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
        gridSize = ((xyz_max - xyz_min) / voxel_size).long().tolist()    
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize)#.to(self.device)
        print('Fine Voxel GridSize', self.gridSize)


        self.app_dim = app_dim
        self.app_n_comp = app_n_comp#[48,12,12]
        # self.gridSize = [164*4, 167*4, 76*4]
        # self.aabbSize = self.aabb[1] - self.aabb[0]
        # self.invaabbSize = 2.0/self.aabbSize

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]
        self.reg = TVLoss()


        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False)#.to(device)

    def init_one_svd(self, n_component, gridSize, scale):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        # return plane_coef, line_coef
        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)
    


    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars_vol = list(self.app_line)+ list(self.app_plane)
        grad_vars_net = list(self.basis_mat.parameters())+list(self.color_net.parameters())+list(self.sigma_net.parameters())
        return grad_vars_vol, grad_vars_net



        
    def TV_loss_app(self):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + self.reg(self.app_plane[idx]) * 1e-2 + self.reg(self.app_line[idx]) * 1e-3
        return total



    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)




    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, is_train=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            feature_map: [num_rays, 3]. Estimated feature sum of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        def raw2alpha(raw_, dists_, act_fn):
            alpha_ = - torch.exp(-act_fn(raw_) * dists_) + 1.
            return torch.cat([alpha_, torch.ones_like(alpha_[:, 0:1])], dim=-1)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]
        # dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # feature = torch.relu(raw[..., 1:])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., :-1, 0]) * raw_noise_std


        density = torch.relu(raw[..., :-1, 0] + noise)
        # print(density.shape, raw.shape)
        if not is_train and self.render_rmnearplane > 0:
            mask = z_vals[:, 1:]
            mask = mask > self.render_rmnearplane / 128
            mask = mask.type_as(density)
            density = mask * density

        # print(density.shape, dists.shape)
        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)

        # alpha = raw2alpha(raw[..., :-1, 3] + noise, dists, act_fn=self.sigma_activate)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        rgb_map = torch.sum(weights[..., None] * raw[..., 1:], -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)


        # mask = weights.sum(-1) > 0.5
        # entropy = Categorical(probs = weights+1e-5).entropy()
        # sparsity_loss = entropy * mask

        return rgb_map, density, acc_map, weights, depth_map#, sparsity_loss

    def sample(self, pts):
        xyz_sampled = (pts.reshape(-1,3)-self.aabb[0]) * self.invaabbSize - 1
        return self.compute_appfeature(xyz_sampled).reshape(pts.shape[0],pts.shape[1],-1)


    def forward(self, pts, viewdirs, fts, pts_embed, dirs_embed, z_vals, rays_d, raw_noise_std, is_train):


        # time1 = time.time()
        input_locs = torch.reshape(pts, [-1, pts.shape[-1]])
        input_locs = pts_embed(input_locs)
        input_dirs = viewdirs[:, None].expand(pts.shape)
        input_dirs = torch.reshape(input_dirs, [-1, viewdirs.shape[-1]])
        input_dirs = dirs_embed(input_dirs)

        # time2 = time.time()

        # sigma
        h = torch.cat([fts.reshape(pts.shape[0]*pts.shape[1],-1),input_locs],-1)
        # h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = h[...,[0]].reshape(pts.shape[0], pts.shape[1], -1)
        # color = torch.zeros((*pts.shape[:2], 3), device=pts.device)
        h = torch.cat([h[...,1:],input_dirs],-1)#.reshape(pts.shape[0], pts.shape[1], -1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        color = torch.sigmoid(h).reshape(pts.shape[0], pts.shape[1], -1)

        color, density_map, acc_map, weights, depth_map = self.raw2outputs(torch.cat([sigma,color],-1), z_vals, rays_d, raw_noise_std, is_train=is_train)

        # time5 = time.time()

        # print(f"Time| embed: {time2-time1:.5f}, sigma: {time3-time2:.5f} raw2output: {time4-time3:.5f}, color: {time5-time4:.5f}")
        return color, depth_map, acc_map, weights


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i + (HALF_PIX - K[0][2])) / K[0][0], -(j + (HALF_PIX - K[1][2])) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i + (HALF_PIX - K[0][2])) / K[0][0], -(j + (HALF_PIX - K[1][2])) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    See Paper supplementary for details
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    # if pytest:
    #     np.random.seed(0)
    #     new_shape = list(cdf.shape[:-1]) + [N_samples]
    #     if det:
    #         u = np.linspace(0., 1., N_samples)
    #         u = np.broadcast_to(u, new_shape)
    #     else:
    #         u = np.random.rand(*new_shape)
    #     u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def smart_load_state_dict(model: nn.Module, state_dict: dict):
    if "network_fn_state_dict" in state_dict.keys():
        state_dict_fn = {k.lstrip("module."): v for k, v in state_dict["network_fn_state_dict"].items()}
        state_dict_fn = {"mlp_coarse." + k: v for k, v in state_dict_fn.items()}

        state_dict_fine = {k.lstrip("module."): v for k, v in state_dict["network_fine_state_dict"].items()}
        state_dict_fine = {"mlp_fine." + k: v for k, v in state_dict_fine.items()}
        state_dict_fn.update(state_dict_fine)
        state_dict = state_dict_fn
    # elif "network_state_dict" in state_dict.keys():
        # state_dict = {k[7:]: v for k, v in state_dict["network_state_dict"].items()}
    else:
        state_dict = state_dict

    # if isinstance(model, nn.DataParallel):
        # state_dict = {"module." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict["network_state_dict"])
