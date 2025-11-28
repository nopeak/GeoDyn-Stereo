import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler


class Combined_Geo_Encoding_Volume(nn.Module):
    def __init__(self, args, cv_chan, geo_volume2, init_fmap1, init_fmap2):
        super(Combined_Geo_Encoding_Volume, self).__init__()
        self.args = args
        self.cv_chan = cv_chan # D' in the paper, number of channels for geometric encoding volume
        
        self.geo_volume2 = geo_volume2 # This is an instance of GeoEncoder
        self.init_fmap1 = init_fmap1   # This is an nn.Conv2d module
        self.init_fmap2 = init_fmap2   # This is an nn.Conv2d module

        # GeoEncoder (geo_volume2) outputs 96 channels.
        # We apply it to 3 different geometric cost volumes, concatenate, then reduce to 3 channels for weights.
        geo_encoder_output_channels = 96 # Output channels of GeoEncoder
        self.selective_weights_head = nn.Conv2d(geo_encoder_output_channels * 3, 3, kernel_size=1, padding=0)

    def _build_single_geometric_corr(self, feat_left, feat_right, corr_radius):
        """
        Builds a single D-channel correlation volume.
        Output shape: [B, D, H, W], where D = 2 * corr_radius + 1.
        """
        B, C, H, W = feat_left.shape
        D_geom = 2 * corr_radius + 1
        
        # Normalize features
        feat_left_norm = feat_left / (torch.norm(feat_left, dim=1, keepdim=True) + 1e-7)
        feat_right_norm = feat_right / (torch.norm(feat_right, dim=1, keepdim=True) + 1e-7)
        
        single_corr_volume = torch.zeros(B, D_geom, H, W, device=feat_left.device, dtype=feat_left.dtype)
        
        for i in range(D_geom):
            disp = i - corr_radius
            shifted_feat_right = torch.zeros_like(feat_right_norm)
            if disp < 0: # Shift right feature to the left
                shifted_feat_right[:, :, :, :W+disp] = feat_right_norm[:, :, :, -disp:]
            elif disp > 0: # Shift right feature to the right
                shifted_feat_right[:, :, :, disp:] = feat_right_norm[:, :, :, :W-disp]
            else: # No shift
                shifted_feat_right = feat_right_norm
            
            # Compute correlation for this disparity
            correlation_slice = torch.sum(feat_left_norm * shifted_feat_right, dim=1) # Shape: [B, H, W]
            single_corr_volume[:, i, :, :] = correlation_slice
            
        return single_corr_volume

    def build_cost_volume_geom(self, feat_l1, feat_r1):
        """
        Builds a list of three geometric cost volumes.
        Each volume is [B, D, H, W], where D = 2 * args.corr_radius + 1.
        """
        # Use the correlation radius from args
        # Note: args.corr_radius defines the depth of the cost volumes (number of disparity planes)
        corr_radius = self.args.corr_radius 

        # For now, all three geometric volumes are constructed using the same method
        # and the same input features (1/4 scale).
        # Future improvements could involve different transformations or features for each.
        geom_cv0 = self._build_single_geometric_corr(feat_l1, feat_r1, corr_radius)
        geom_cv1 = self._build_single_geometric_corr(feat_l1, feat_r1, corr_radius)
        geom_cv2 = self._build_single_geometric_corr(feat_l1, feat_r1, corr_radius)

        return [geom_cv0, geom_cv1, geom_cv2]

    @staticmethod
    def corr(fmap1_tensor, fmap2_tensor, radius=4): # fmap1_tensor, fmap2_tensor are actual tensors
        B, D_cv, H, W = fmap1_tensor.shape # D_cv is the number of channels from init_fmap modules
        # radius here is for the correlation operation itself, distinct from args.corr_radius for cost volume depth
        # This 'corr' computes a correlation map, not a full cost volume like build_corr
        
        # Normalize features (optional, but common in correlation)
        fmap1_norm = fmap1_tensor / (torch.norm(fmap1_tensor, p=2, dim=1, keepdim=True) + 1e-8)
        fmap2_norm = fmap2_tensor / (torch.norm(fmap2_tensor, p=2, dim=1, keepdim=True) + 1e-8)

        corr_maps = []
        for i in range(-radius, radius + 1):
            shifted_fmap2 = torch.zeros_like(fmap2_norm)
            if i < 0: # Shift right
                shifted_fmap2[:, :, :, :W+i] = fmap2_norm[:, :, :, -i:]
            elif i > 0: # Shift left
                shifted_fmap2[:, :, :, i:] = fmap2_norm[:, :, :, :W-i]
            else:
                shifted_fmap2 = fmap2_norm
            
            corr = torch.sum(fmap1_norm * shifted_fmap2, dim=1, keepdim=True) # [B, 1, H, W]
            corr_maps.append(corr)
        
        # Stack to form a [B, (2*radius+1), H, W] correlation map
        # This is the 'init_corr' used in the paper, often with D_cv = (2*radius+1) channels
        init_corr_volume = torch.cat(corr_maps, dim=1)
        return init_corr_volume

    def forward(self, feat_l1, feat_r1, cost_volume_geom_list):
        # feat_l1, feat_r1 are the 1/4 scale image features [B, C_img_feat, H, W]
        # cost_volume_geom_list is a list of 3 raw geometric correlation volumes [B, D_raw_geom, H, W]
        # D_raw_geom is typically (2 * args.corr_radius + 1)

        # 1. Calculate init_corr_val
        # Apply the nn.Conv2d modules to get tensors
        fmap1_tensor = self.init_fmap1(feat_l1) # Output: [B, D_channels_from_init_fmap, H, W]
        fmap2_tensor = self.init_fmap2(feat_r1) # Output: [B, D_channels_from_init_fmap, H, W]
        
        # The self.corr method expects D_channels_from_init_fmap to be the depth of the correlation it computes.
        # Let's assume init_fmap1/2 output D_cv channels, and self.corr computes a correlation of depth D_cv.
        # The original paper's init_corr has D' channels (self.cv_chan).
        # The self.corr method as defined above will output 2*radius+1 channels.
        # We need to ensure this matches self.cv_chan or is sliced.
        # Let's assume self.corr's radius is chosen such that its output depth is self.cv_chan.
        # For example, if self.cv_chan = 9, then radius for self.corr should be 4.
        # Or, init_fmap1/2 output self.cv_chan channels, and self.corr uses these directly.

        # Let's redefine self.corr to be more aligned with how init_corr is typically formed
        # if init_fmap1/2 already produce the desired D' channels.
        # If init_fmap1/2 output D' channels, then init_corr is just their correlation.
        # The `corr` method above computes a multi-displacement correlation.
        # If init_fmap1 and init_fmap2 output D_cv channels, and D_cv == self.cv_chan,
        # then init_corr_val = torch.sum(fmap1_tensor * fmap2_tensor, dim=1, keepdim=True) is too simple.
        # The `corr` method as written is more plausible for `init_corr`.
        # Let radius for self.corr be such that 2*radius+1 = self.cv_chan
        corr_radius_for_init_corr = (self.cv_chan - 1) // 2
        init_corr_val = self.corr(fmap1_tensor, fmap2_tensor, radius=corr_radius_for_init_corr) # [B, self.cv_chan, H, W]


        # 2. Calculate selective_weights_val
        # Each item in cost_volume_geom_list is [B, D_raw_geom, H, W]
        g0 = self.geo_volume2(cost_volume_geom_list[0]) # Output: [B, 96, H, W]
        g1 = self.geo_volume2(cost_volume_geom_list[1]) # Output: [B, 96, H, W]
        g2 = self.geo_volume2(cost_volume_geom_list[2]) # Output: [B, 96, H, W]
        
        combined_g = torch.cat([g0, g1, g2], dim=1) # Output: [B, 96*3, H, W]
        selective_weights_logits = self.selective_weights_head(combined_g) # Output: [B, 3, H, W]
        selective_weights_val = torch.softmax(selective_weights_logits, dim=1)


        # 3. Construct final_cost_volume
        # Slice the raw geometric cost volumes to self.cv_chan if their depth D_raw_geom > self.cv_chan
        # D_raw_geom is (2 * self.args.corr_radius + 1)
        # self.cv_chan is args.cost_volume_planes
        # actual_geo_feat_channels = min(D_raw_geom, self.cv_chan)
        # The BasicMultiUpdateBlock uses actual_geo_feat_channels for its GeoEncoders.
        # Here, we are constructing the *final* cost volume for the initial disparity estimation,
        # which is often init_corr + weighted sum of geometric features.
        
        cv_geom0_sliced = cost_volume_geom_list[0][:, :self.cv_chan, :, :]
        cv_geom1_sliced = cost_volume_geom_list[1][:, :self.cv_chan, :, :]
        cv_geom2_sliced = cost_volume_geom_list[2][:, :self.cv_chan, :, :]

        weighted_geom_cv = selective_weights_val[:,0:1] * cv_geom0_sliced + \
                           selective_weights_val[:,1:2] * cv_geom1_sliced + \
                           selective_weights_val[:,2:3] * cv_geom2_sliced
        
        final_cost_volume = init_corr_val + weighted_geom_cv # [B, self.cv_chan, H, W]

        return final_cost_volume, init_corr_val, selective_weights_val

    # Remove the extraneous __call__ method below
    # def __call__(self, disp, coords):
    #     r = self.radius
    #     b, _, h, w = disp.shape
    #     init_corr_pyramid = []
    #     geo_feat0_pyramid = []
    #     dx = torch.linspace(-r, r, 2*r+1)
    #     dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)

    #     x1 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2
    #     y0 = torch.zeros_like(x1)
    #     disp_lvl1 = torch.cat([x1, y0], dim=-1)
    #     geo_feat1 = bilinear_sampler(self.geo_volume1, disp_lvl1)
    #     geo_feat1 = geo_feat1.view(b, h, w, -1)

    #     x2 = dx + disp.reshape(b*h*w, 1, 1, 1) / 4
    #     y0 = torch.zeros_like(x2)
    #     disp_lvl2 = torch.cat([x2, y0], dim=-1)
    #     geo_feat2 = bilinear_sampler(self.geo_volume2, disp_lvl2)
    #     geo_feat2 = geo_feat2.view(b, h, w, -1)

    #     for i in range(self.num_levels):
    #         geo_volume0 = self.geo_volume0_pyramid[i]
    #         x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
    #         y0 = torch.zeros_like(x0)
    #         disp_lvl0 = torch.cat([x0,y0], dim=-1)
    #         geo_feat0 = bilinear_sampler(geo_volume0, disp_lvl0)
    #         geo_feat0 = geo_feat0.view(b, h, w, -1)
    #         geo_feat0_pyramid.append(geo_feat0)

    #         init_corr = self.init_corr_pyramid[i]
    #         init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
    #         init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
    #         init_corr = bilinear_sampler(init_corr, init_coords_lvl)
    #         init_corr = init_corr.view(b, h, w, -1)
    #         init_corr_pyramid.append(init_corr)

    #     init_corr = torch.cat(init_corr_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous().float()
    #     geo_feat0 = torch.cat(geo_feat0_pyramid, dim=-1)
    #     geo_feat0 = geo_feat0.permute(0, 3, 1, 2).contiguous().float()
    #     geo_feat1 = geo_feat1.permute(0, 3, 1, 2).contiguous().float()
    #     geo_feat2 = geo_feat2.permute(0, 3, 1, 2).contiguous().float()

    #     return geo_feat0, geo_feat1, geo_feat2, init_corr
 
    # Remove the duplicate static method corr below
    # @staticmethod
    # def corr(fmap1, fmap2):
    #     B, D, H, W1 = fmap1.shape
    #     _, _, _, W2 = fmap2.shape
    #     fmap1 = fmap1.view(B, D, H, W1)
    #     fmap2 = fmap2.view(B, D, H, W2)
    #     corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
    #     corr = corr.reshape(B, H, W1, 1, W2).contiguous()
    #     return corr