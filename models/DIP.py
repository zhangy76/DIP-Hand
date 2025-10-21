import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.meshconv_utils import SpiralEnblock, spiral_tramsform
from models.utils import rot6d_to_rotmat


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class MeshConv(nn.Module):
    def __init__(self, device, latent_channels=128, in_channels=3):
        super().__init__()

        template_fp = "./assets/MANO/MeshConv_template.ply"
        transform_fp = "./assets/MANO/MeshConv_transform.pkl"
        ds_factors = [2, 2, 2, 2]
        seq_length = [9, 9, 9, 9]
        dilation = [1, 1, 1, 1]
        self.spiral_indices, self.down_transform, self.up_transform, _ = (
            spiral_tramsform(
                transform_fp, template_fp, ds_factors, seq_length, dilation
            )
        )

        out_channels = [32, 64, 64, 64]
        num_vert = self.down_transform[-1].size(0)

        self.in_channels = in_channels
        self.conv_layer = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.conv_layer.append(
                    SpiralEnblock(
                        in_channels,
                        out_channels[idx],
                        self.spiral_indices[idx].to(device),
                    )
                )
            else:
                self.conv_layer.append(
                    SpiralEnblock(
                        out_channels[idx - 1],
                        out_channels[idx],
                        self.spiral_indices[idx].to(device),
                    )
                )
        self.conv_layer.append(nn.Linear(num_vert * out_channels[-1], latent_channels))

        self.output_dim = latent_channels

    def forward(self, x):
        batch_size = x.shape[1]
        x = x.reshape([-1, 778, self.in_channels])
        for i, layer in enumerate(self.conv_layer):
            if i != len(self.conv_layer) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x.reshape([-1, batch_size, self.output_dim])


class MLP(nn.Module):
    def __init__(self, dims=(128, 128), activation="gelu", norm_type=None):
        super().__init__()
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "sigmoid":
            self.activation = torch.sigmoid

        self.num_dims = len(dims)
        self.norm_type = norm_type
        self.affine_layers = nn.ModuleList()

        for i in range(self.num_dims - 1):
            self.affine_layers.append(nn.Linear(dims[i], dims[i + 1]))

    def forward(self, x):
        for i, affine in enumerate(self.affine_layers):
            x = affine(x)
            if i == (len(self.affine_layers) - 1):
                continue
            x = self.activation(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DIP(nn.Module):

    def __init__(
        self,
        device,
        seqlen,
        decoder: bool,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        n_shape = 10
        n_pose = 16 * 6
        n_state = 4
        self.decoder = decoder
        self.d_model = d_model
        self.device = device
        self.seqlen = seqlen

        # transformer
        self.start = torch.nn.Parameter(
            torch.rand([1, 1, d_model]).to(device) * 0.01, requires_grad=True
        )
        self.src_mask = generate_square_subsequent_mask(seqlen).to(device)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # diffusion step encoder
        self.n_step_encoder = MLP(
            dims=(
                d_model,
                d_model,
                d_model,
            )
        )

        # Transformer Encoder
        self.embedding_encoder = MeshConv(
            device, latent_channels=d_model, in_channels=6
        )
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, activation, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Transformer Dencoder
        self.embedding_decoder = MLP(
            dims=(
                d_model + n_state,
                d_model,
            )
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers)

        # Regression Layer
        self.shape_decoder = MLP(dims=(d_model, d_model, n_shape))
        self.pose_decoder = MLP(dims=(d_model, d_model * 2, n_pose))
        self.state_decoder = MLP(dims=(d_model, d_model, n_state))

    def forward(
        self,
        y0_vertices,
        xn_vertices,
        n_step,
        src_mask,
        x0_vertices,
        state,
        mano,
    ) -> Tensor:
        """
        Args:
            y0_vertices: Tensor, shape [batch_size, seq_len, 778, 3]
            xn_vertices: Tensor, shape [batch_size, seq_len, 778, 3]
            n_step: Tensor, shape [batch_size]
            src_mask: Tensor, shape [batch_size, seq_len]
            x0_vertices: Tensor, shape [batch_size, seq_len, 778, 3]
            state: Tensor, shape [batch_size, 4]
        """
        N, T, _, _ = xn_vertices.shape

        # masking
        if src_mask is None:
            src_mask = torch.ones((N, T), dtype=torch.bool).to(self.device)
        src_mask_padding = ~src_mask.clone()
        src_mask_padding = torch.cat(
            [torch.zeros((N, 1), dtype=torch.bool).to(self.device), src_mask_padding],
            dim=1,
        )

        # diffusion step embedding
        n_embedding = self.n_step_encoder(
            self.pos_encoder.pe[0, n_step.detach().cpu().numpy().tolist()]
        ).unsqueeze(1)

        # transformer encoder
        xn_input = torch.cat([y0_vertices, xn_vertices], dim=3)
        xn_embedding = self.embedding_encoder(xn_input)
        n_xn_embedding = torch.cat([n_embedding, xn_embedding], dim=1)
        n_xn_embedding = self.pos_encoder(n_xn_embedding)
        z_embedding = self.transformer_encoder(
            n_xn_embedding, src_key_padding_mask=src_mask_padding
        )[:, 1:]  # N,T,D

        # encoder prediction
        x0_shape_encoder_pred = self.shape_decoder(z_embedding[:, :1])
        x0_pose_encoder_pred = self.pose_decoder(z_embedding)
        state_encoder_pred = self.state_decoder(z_embedding)

        x0_shape_encoder_pred_t = x0_shape_encoder_pred.expand(
            -1, T, -1
        ).reshape(-1, 10)
        x0_rotmat_encoder_pred_t = rot6d_to_rotmat(
            x0_pose_encoder_pred.reshape([-1, 6])
        ).view(-1, 16, 3, 3)
        x0_mano_encoder_pred_t = mano.forward(
            betas=x0_shape_encoder_pred_t, thetas_rotmat=x0_rotmat_encoder_pred_t
        )

        # transformer decoder
        if self.decoder:
            x0_encoder_pred = torch.cat(
                [y0_vertices, x0_mano_encoder_pred_t.vertices.reshape([-1, T, 778, 3])],
                dim=3,
            )

            state_pred = F.gumbel_softmax(state_encoder_pred, tau=0.1, hard=True)

            x0_embedding = torch.cat(
                [
                    self.start.repeat(N, 1, 1),
                    self.embedding_decoder(
                        torch.cat(
                            [
                                self.embedding_encoder(x0_encoder_pred[:, :-1]),
                                state_pred[:, :-1],
                            ],
                            dim=2,
                        )
                    ),
                ],
                dim=1,
            )

            x0_embedding = self.pos_encoder(x0_embedding)

            z_embedding_decoder = self.transformer_decoder(
                tgt=x0_embedding, memory=z_embedding, tgt_mask=self.src_mask
            )

            x0_shape_decoder_pred = self.shape_decoder(z_embedding_decoder[:, :1])
            x0_pose_decoder_pred = self.pose_decoder(z_embedding_decoder)
            state_decoder_pred = self.state_decoder(z_embedding_decoder)

            x0_shape_decoder_pred_t = x0_shape_decoder_pred.expand(
                -1, T, -1
            ).reshape(-1, 10)
            x0_rotmat_decoder_pred_t = rot6d_to_rotmat(
                x0_pose_decoder_pred.reshape([-1, 6])
            ).view(-1, 16, 3, 3)
            x0_mano_decoder_pred_t = mano.forward(
                betas=x0_shape_decoder_pred_t, thetas_rotmat=x0_rotmat_decoder_pred_t
            )
        else:
            x0_shape_decoder_pred = x0_shape_encoder_pred
            x0_pose_decoder_pred = x0_pose_encoder_pred
            state_decoder_pred = state_encoder_pred

        # prediction
        encoder_pred = (
            x0_shape_decoder_pred,
            x0_pose_encoder_pred,
            state_encoder_pred,
            x0_shape_encoder_pred_t,
            x0_rotmat_encoder_pred_t,
            x0_mano_encoder_pred_t,
        )
        decoder_pred = (
            x0_shape_encoder_pred,
            x0_pose_decoder_pred,
            state_decoder_pred,
            x0_shape_decoder_pred_t,
            x0_rotmat_decoder_pred_t,
            x0_mano_decoder_pred_t,
        )

        return encoder_pred, decoder_pred
