import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from core.convolutional_graph_ssm.classification.models.graph_ssm import (
    GraphSSM as ConvGraphSSM,
)
from core.graph_ssm.main import GraphSSM as TemporalGraphSSM


class Baseline_3ET(nn.Module):
    def __init__(self, height, width, input_dim=1):
        super(Baseline_3ET, self).__init__()

        # 1) 2D (spatial) GraphSSM
        #    We'll do 2 levels, 2 blocks each, base channels=16
        self.spatial_backbone = ConvGraphSSM(
            in_chans=input_dim,
            num_levels=2,
            depths=[2, 2],
            channels=16,
            mlp_ratio=4.0,
            drop_path_rate=0.0,
            drop_rate=0.0,
            one_layer=False,
            two_layer=False,
        )
        # The final dimension of the spatial backbone:
        self.d_model = (
            self.spatial_backbone.num_features
        )  # typically 16 * (2^(2-1)) = 32

        # 2) Temporal GraphSSM
        self.temporal_ssm = TemporalGraphSSM(
            d_model=self.d_model, d_state=16, d_conv=4, expand=2
        )

        # 3) Final MLP: we want to predict (x, y) => dimension=2
        self.fc_out = nn.Linear(self.d_model, 2)

        # For classification
        # self.fc_out = nn.Linear(self.d_model, 11)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
            B=batch, T=sequence length, C=input_dim, H=height, W=width
        """
        B, T, C, H, W = x.shape

        # (A) Flatten time into batch for the 2D GraphSSM
        x_2d = x.view(B * T, C, H, W)  # => [B*T, C, H, W]

        # Pass through spatial GraphSSM => [B*T, d_model, H', W']
        feat_2d = self.spatial_backbone(x_2d)
        # We do global average pooling to get a single vector per frame
        if feat_2d.dim() == 4:
            # shape [B*T, d_model, H', W']
            feat_2d = F.adaptive_avg_pool2d(feat_2d, (1, 1))  # => [B*T, d_model, 1, 1]
            feat_2d = feat_2d.view(B * T, self.d_model)  # => [B*T, d_model]

        # (B) Reshape into a sequence => [B, T, d_model]
        seq_in = feat_2d.view(B, T, self.d_model)

        # Forward pass through temporal GraphSSM => [B, T, d_model]
        seq_out = self.temporal_ssm(seq_in, context_len=T)

        # Final linear => [B, T, 2]
        coords = self.fc_out(seq_out)
        return coords


if __name__ == "__main__":
    # Example usage
    # -----------------------
    # Create a Baseline_3ET model for 64x64 input images with 1 input channel
    model = Baseline_3ET(height=64, width=64, input_dim=1)

    # Generate a random batch of data: B=2, T=5 frames, C=1 channel, H=64, W=64
    B, T, C, H, W = 2, 5, 1, 64, 64
    x = torch.randn(B, T, C, H, W)

    # Forward pass
    output = model(x)
    print("Output shape:", output.shape)  # should be [B, T, 2]
