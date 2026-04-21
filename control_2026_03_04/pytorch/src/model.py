# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Source: https://huggingface.co/JensLundsgaard/control-2026-03-04
"""
ConvLSTM Autoencoder

Complete high-quality ConvLSTM Autoencoder with:
- 2D CNN spatial compression with ResNet-style residual connections
- LSTM temporal modeling over flattened spatial latents
- ConvTranspose spatial reconstruction with residual up-blocks
- Compatible with HuggingFace Hub via PyTorchModelHubMixin
- Works with 128x128 input images
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class ResidualBlock(nn.Module):
    """Residual block for encoder with optional downsampling."""

    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()

        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResidualUpBlock(nn.Module):
    """Residual block for decoder with upsampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.upsample(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    """2D CNN spatial compression + LSTM temporal modeling over flattened latents."""

    def __init__(
        self,
        input_channels=1,
        hidden_dim=256,
        num_layers=2,
        latent_size=4096,
        use_convlstm=True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_size = latent_size

        # 128 -> 64 -> 32 -> 16
        self.spatial_cnn = nn.Sequential(
            ResidualBlock(input_channels, 64, downsample=True),
            ResidualBlock(64, 128, downsample=True),
            ResidualBlock(128, 256, downsample=True),
        )

        self.use_convlstm = use_convlstm
        if self.use_convlstm:
            self.lstm_enc = nn.LSTM(latent_size, latent_size, batch_first=True)
        else:
            self.lstm_enc = None

        self.dropout = nn.Dropout(0.1)

        self.latent_compress = nn.Linear(hidden_dim * 16 * 16, latent_size)
        self.lin1 = nn.Linear(latent_size, latent_size)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        if H != 128 or W != 128:
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=True)

        x = self.spatial_cnn(x)
        _, C2, H2, W2 = x.shape
        x = x.view(B, T, C2, H2, W2)

        h_seq = x

        B, T, C, H, W = h_seq.shape
        h_flat = h_seq.view(B, T, C * H * W)
        z_compressed = self.dropout(h_flat)
        z_compressed = F.relu(self.latent_compress(z_compressed))
        if self.use_convlstm:
            z_compressed, _ = self.lstm_enc(z_compressed)
        z_compressed = F.relu(self.lin1(z_compressed))
        z_seq = z_compressed.view(B, T, self.latent_size)

        return z_seq


class Decoder(nn.Module):
    """Linear expansion + LSTM temporal decoding + ConvTranspose spatial reconstruction."""

    def __init__(
        self,
        seq_len,
        latent_size=4096,
        latent_dim=256,
        hidden_dim=256,
        num_layers=2,
        use_convlstm=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.latent_size = latent_size

        self.latent_expand = nn.Linear(latent_size, latent_dim * 16 * 16)

        self.use_convlstm = use_convlstm
        if self.use_convlstm:
            self.lstm_dec = nn.LSTM(latent_size, latent_size, batch_first=True)
        else:
            self.lstm_dec = None

        # 16 -> 32 -> 64 -> 128
        self.spatial_decoder = nn.Sequential(
            ResidualUpBlock(hidden_dim, 128),
            ResidualUpBlock(128, 64),
            ResidualUpBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.lin1 = nn.Linear(latent_size, latent_size)

    def forward(self, z_seq):
        B, T, L = z_seq.shape

        z_flat = z_seq
        z_flat = F.relu(self.lin1(z_flat))
        if self.use_convlstm:
            z_flat, _ = self.lstm_dec(z_flat)
        z_expanded = F.relu(self.latent_expand(z_flat))
        z_spatial = z_expanded.view(B, T, self.latent_dim, 16, 16)

        h_seq = z_spatial
        B, T, C, H, W = h_seq.shape
        h_seq = h_seq.view(B * T, C, H, W)
        x_rec = self.spatial_decoder(h_seq)
        x_rec = x_rec.view(B, T, 1, 128, 128)

        return x_rec


class ConvLSTMAutoencoder(nn.Module, PyTorchModelHubMixin):
    """Complete ConvLSTM Autoencoder compatible with HuggingFace Hub."""

    def __init__(
        self,
        config=None,
        seq_len=20,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=4096,
        use_classifier=True,
        num_classes=2,
        use_latent_split=False,
        dropout_rate=0.1,
        use_convlstm=True,
        use_residual=True,
        use_batchnorm=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.use_classifier = use_classifier
        self.encoder_hidden_dim = encoder_hidden_dim
        self.latent_size = latent_size
        self.use_latent_split = use_latent_split
        self.dropout_rate = dropout_rate
        self.use_convlstm = use_convlstm
        self.use_residual = use_residual
        self.use_batchnorm = use_batchnorm

        if config is not None:
            if isinstance(config, dict):
                self.seq_len = config.get("seq_len", seq_len)
                self.use_classifier = config.get("use_classifier", use_classifier)
                self.encoder_hidden_dim = config.get(
                    "encoder_hidden_dim", encoder_hidden_dim
                )
                self.latent_size = config.get("latent_size", latent_size)
                self.use_latent_split = config.get("use_latent_split", use_latent_split)
                self.dropout_rate = config.get("dropout_rate", dropout_rate)
                self.use_convlstm = config.get("use_convlstm", use_convlstm)
                self.use_residual = config.get("use_residual", use_residual)
                self.use_batchnorm = config.get("use_batchnorm", use_batchnorm)
            else:
                self.seq_len = config.seq_len
                self.use_classifier = config.use_classifier
                self.encoder_hidden_dim = config.encoder_hidden_dim
                self.latent_size = config.latent_size
                self.use_latent_split = config.use_latent_split
                self.dropout_rate = config.dropout_rate
                self.use_convlstm = config.use_convlstm
                self.use_residual = config.use_residual
                self.use_batchnorm = config.use_batchnorm

        self.encoder = Encoder(
            latent_size=self.latent_size,
            use_convlstm=self.use_convlstm,
        )

        self.decoder = Decoder(
            seq_len=self.seq_len,
            latent_size=self.latent_size,
            use_convlstm=self.use_convlstm,
        )

    def forward(self, x):
        B, T, C, orig_H, orig_W = x.shape

        z_seq = self.encoder(x)
        x_rec = self.decoder(z_seq)

        if orig_H != 128 or orig_W != 128:
            x_rec_flat = x_rec.view(B * T, C, 128, 128)
            x_rec_flat = F.interpolate(
                x_rec_flat, size=(orig_H, orig_W), mode="bilinear", align_corners=True
            )
            x_rec = x_rec_flat.view(B, T, C, orig_H, orig_W)

        return x_rec, z_seq
