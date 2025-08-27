import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
import yaml
from easydict import EasyDict
import os, sys
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms as T

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from tree_scanning import Tree_SSM


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return EasyDict(config)


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(
    dim, norm_layer, in_format="channels_last", out_format="channels_last", eps=1e-6
):
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == "channels_last":
            layers.append(to_channels_last())
    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first":
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f"build_norm_layer does not support {norm_layer}")
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    elif act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    elif act_layer == "GELU":
        return nn.GELU()

    raise NotImplementedError(f"build_act_layer does not support {act_layer}")


class StemLayer(nn.Module):
    r"""Stem layer of GraphSSM
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self, in_chans=3, out_chans=96, act_layer="GELU", norm_layer="BN"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_chans, out_chans // 2, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = build_norm_layer(
            out_chans // 2, norm_layer, "channels_first", "channels_first"
        )
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(
            out_chans // 2, out_chans, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = build_norm_layer(
            out_chans, norm_layer, "channels_first", "channels_last"
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    r"""Downsample layer of GraphSSM
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer="LN"):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, 2 * channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.norm = build_norm_layer(
            2 * channels, norm_layer, "channels_first", "channels_last"
        )

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)
        return x


class MLPLayer(nn.Module):
    r"""MLP layer of GraphSSM
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="GELU",
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GraphSSMLayer(nn.Module):
    def __init__(
        self,
        channels,
        mlp_ratio=4.0,
        drop=0.0,
        norm_layer="LN",
        drop_path=0.0,
        act_layer="GELU",
        post_norm=False,
        layer_scale=None,
        with_cp=False,
    ):
        super().__init__()
        self.channels = channels
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, "LN")
        self.post_norm = post_norm
        self.TreeSSM = Tree_SSM(
            d_model=channels,
            d_state=1,
            ssm_ratio=2,
            ssm_rank_ratio=2,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # ==========================
            d_conv=3,
            conv_bias=False,
            # ==========================
            dropout=0.0,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = build_norm_layer(channels, "LN")
        self.mlp = MLPLayer(
            in_features=channels,
            hidden_features=int(channels * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(channels), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(channels), requires_grad=True
            )

    def forward(self, x):
        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.TreeSSM(x)))
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
                else:
                    x = x + self.drop_path(self.TreeSSM(self.norm1(x)))
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.TreeSSM(x)))
                x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.TreeSSM(self.norm1(x)))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class GraphSSMBlock(nn.Module):
    def __init__(
        self,
        channels,
        depth,
        downsample=True,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer="GELU",
        norm_layer="LN",
        post_norm=False,
        layer_scale=None,
        with_cp=False,
    ):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm

        self.blocks = nn.ModuleList(
            [
                GraphSSMLayer(
                    channels=channels,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    post_norm=post_norm,
                    layer_scale=layer_scale,
                    with_cp=with_cp,
                )
                for i in range(depth)
            ]
        )
        self.norm = build_norm_layer(channels, "LN")
        self.downsample = (
            DownsampleLayer(channels=channels, norm_layer=norm_layer)
            if downsample
            else None
        )

    def forward(self, x, return_wo_downsample=False):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)
        if return_wo_downsample:
            x_ = x
        if self.downsample is not None:
            x = self.downsample(x)

        if return_wo_downsample:
            return x, x_
        return x


class GraphSSM(nn.Module):
    def __init__(
        self,
        config=None,
        drop_rate=0.0,
        drop_path_type="linear",
        layer_scale=None,
        post_norm=False,
        with_cp=False,
        one_layer=False,
        two_layer=False,
        num_levels=None,
        depths=None,
        channels=None,
        mlp_ratio=None,
        drop_path_rate=None,
        in_chans=3,
        **kwargs,
    ):
        super().__init__()

        if config is not None:
            self.num_levels = len(config.MODEL.CONV_GRAPH_SSM.DEPTHS)
            self.depths = config.MODEL.CONV_GRAPH_SSM.DEPTHS
            self.channels = config.MODEL.CONV_GRAPH_SSM.CHANNELS
            self.mlp_ratio = config.MODEL.CONV_GRAPH_SSM.MLP_RATIO
            self.drop_path_rate = config.MODEL.DROP_PATH_RATE
        else:
            self.num_levels = num_levels
            self.depths = depths
            self.channels = channels
            self.mlp_ratio = mlp_ratio
            self.drop_path_rate = drop_path_rate

        self.act_layer = "GELU"
        self.norm_layer = "LN"
        self.drop_rate = drop_rate
        self.one_layer = one_layer
        self.two_layer = two_layer
        self.in_chans = in_chans

        self.num_features = int(self.channels * 2 ** (self.num_levels - 1))

        print(f"using core type: tree_scanning_algorithm")
        print(f"using activation layer: {self.act_layer}")
        print(f"using main norm layer: {self.norm_layer}")
        print(f"using dpr: {drop_path_type}, {self.drop_path_rate}")

        self.patch_embed = StemLayer(
            in_chans=self.in_chans,
            out_chans=self.channels,
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
        )
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
        ]
        if drop_path_type == "uniform":
            for i in range(len(dpr)):
                dpr[i] = self.drop_path_rate

        self.levels = nn.ModuleList()

        # Determine number of levels based on configuration
        num_levels = {"one_layer": 1, "two_layer": 2, "default": self.num_levels}.get(
            "one_layer"
            if self.one_layer
            else "two_layer" if self.two_layer else "default"
        )

        self.num_levels = num_levels

        # Create blocks
        for i in range(self.num_levels):
            level = self._create_ssm_block(
                i, dpr, post_norm, layer_scale, with_cp, one_layer=self.one_layer
            )
            self.levels.append(level)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_layers = len(self.depths)
        self.apply(self._init_weights)

    def _create_ssm_block(
        self, i, dpr, post_norm, layer_scale, with_cp, one_layer=False
    ):
        """Helper method to create a GraphSSMBlock with given parameters"""
        return GraphSSMBlock(
            channels=int(self.channels * 2**i),
            depth=self.depths[i],
            mlp_ratio=self.mlp_ratio,
            drop=self.drop_rate,
            drop_path=dpr[sum(self.depths[:i]) : sum(self.depths[: i + 1])],
            act_layer=self.act_layer,
            norm_layer=self.norm_layer,
            post_norm=post_norm,
            downsample=(i < self.num_levels - 1),
            layer_scale=layer_scale,
            with_cp=with_cp,
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def lr_decay_keywards(self, decay_ratio=0.87):
        lr_ratios = {}

        # blocks
        idx = 0
        for i in range(4):
            layer_num = 3 - i  # 3 2 1 0
            for j in range(self.depths[layer_num]):
                block_num = self.depths[layer_num] - j - 1
                tag = "levels.{}.blocks.{}.".format(layer_num, block_num)
                decay = 1.0 * (decay_ratio**idx)
                lr_ratios[tag] = decay
                idx += 1
        # patch_embed (before stage-1)
        lr_ratios["patch_embed"] = lr_ratios["levels.0.blocks.0."]
        # levels.0.downsample (between stage-1 and stage-2)
        lr_ratios["levels.0.downsample"] = lr_ratios["levels.1.blocks.0."]
        lr_ratios["levels.0.norm"] = lr_ratios["levels.1.blocks.0."]
        # levels.1.downsample (between stage-2 and stage-3)
        lr_ratios["levels.1.downsample"] = lr_ratios["levels.2.blocks.0."]
        lr_ratios["levels.1.norm"] = lr_ratios["levels.2.blocks.0."]
        # levels.2.downsample (between stage-3 and stage-4)
        lr_ratios["levels.2.downsample"] = lr_ratios["levels.3.blocks.0."]
        lr_ratios["levels.2.norm"] = lr_ratios["levels.3.blocks.0."]
        return lr_ratios

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)
            if self.one_layer:
                break  # Exit after the first level if one_layer is True

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        return x

    def forward_features_seq_out(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        seq_out = []
        for level in self.levels:
            x, x_ = level(x, return_wo_downsample=True)
            seq_out.append(x_)
            if self.one_layer:
                break  # Exit after the first level if one_layer is True
        return seq_out

    def forward(self, x):
        # for GraphSSM-T/S/B/L/XL
        x = self.forward_features(x)

        x = x.permute(0, 3, 1, 2)

        return x


def load_partial_weights(model, weights_path):
    state_dict = torch.load(weights_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    model_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"Loaded {len(pretrained_dict)} / {len(model_dict)} layers")


# ----------------------------------------------------
# Visualization Helper
# ----------------------------------------------------
def visualize_feature_map(feature_map, out_dir, layer_idx):
    """
    Saves the first few channels of 'feature_map' (assumed shape [B, H, W, C] or [B, C, H, W])
    as PNG files in the specified output directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    # If channels are last, reorder to BCHW for easier saving:
    # (B, C, H, W)
    if feature_map.dim() == 4 and feature_map.shape[-1] == feature_map.shape[1]:
        # Possibly ambiguous shapes, skipping this check:
        pass
    elif feature_map.shape[1] != 1 and feature_map.shape[1] != 3:
        # shape is (B, H, W, C), permute to (B, C, H, W)
        feature_map = feature_map.permute(0, 3, 1, 2)

    # We'll save first 3 channels as an RGB-like image (or grayscale if less).
    # If you have fewer or more channels, adapt accordingly.
    num_channels = feature_map.shape[1]
    max_channels_to_save = min(num_channels, 3)

    # Clip to 0..1 range or normalize for visualization if needed
    # For now, let's just do a min-max normalization channel-wise:
    for b in range(feature_map.shape[0]):
        fm = feature_map[b]
        for c in range(max_channels_to_save):
            channel_data = fm[c, :, :]
            # Normalize to [0,1] for saving
            cmin, cmax = channel_data.min(), channel_data.max()
            if cmax - cmin > 1e-5:
                channel_data = (channel_data - cmin) / (cmax - cmin)
            else:
                channel_data = channel_data - cmin  # everything would be 0
            fm[c, :, :] = channel_data

        # If we only have 1 channel or 2 channels, we can still broadcast to 3-ch for a quick view:
        if max_channels_to_save < 3:
            # replicate or fill channels to view as an RGB
            fm_3ch = fm[0:1].repeat(3, 1, 1)
        else:
            fm_3ch = fm[:3]

        # Save as a grid. Or you can convert each channel to a separate image.
        save_path = os.path.join(out_dir, f"layer_{layer_idx}_batch_{b}.png")
        vutils.save_image(fm_3ch, save_path)
        print(f"Saved feature map to {save_path}")


if __name__ == "__main__":
    # Example usage. You can parse these from argparse for a real script.

    # update if you have a new config file
    config_path = "core/convolutional_graph_ssm/classification/config/convolutional_graph_ssm_b_1k_224.yaml"
    weights_path = "core/convolutional_graph_ssm/classification/weights/convolutional_graph_ssm_base.pth"  # If you have pretrained weights, specify here

    img_path = "core/convolutional_graph_ssm/classification/cat.jpeg"  # Replace with your input image
    out_dir = "core/convolutional_graph_ssm/classification/output_folder"

    # 1. Load config (optional if you have a real config file)
    if os.path.isfile(config_path):
        config = load_config(config_path)
    else:
        print(f"Config not found at {config_path}, using default settings.")
        config = None

    # 2. Create the model
    model = GraphSSM(
        config=config,
        one_layer=False,  # or True if you want only the first stage
        two_layer=False,  # or True if you want only the first two stages
    ).cuda()

    # 3. (Optional) Load partial weights
    if weights_path is not None and os.path.isfile(weights_path):
        load_partial_weights(model, weights_path)
    else:
        print("No valid weights specified; using random initialization.")

    model.eval()  # set to eval mode

    # 4. Load and preprocess image
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")

    img = Image.open(img_path).convert("RGB")
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),  # 0..1 range
        ]
    )
    x = transform(img).unsqueeze(0).cuda()  # shape: [1, 3, 224, 224]

    # 5. Forward pass + getting intermediate features
    with torch.no_grad():
        # We can get outputs of each stage before downsampling:
        seq_outputs = model.forward_features_seq_out(x)

    # 6. Visualize or save the intermediate outputs
    for idx, feature_map in enumerate(seq_outputs):
        visualize_feature_map(feature_map, out_dir, layer_idx=idx)

    print("Done!")
