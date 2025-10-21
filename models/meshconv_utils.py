import os
import pickle

import numpy as np
import openmesh as om
from sklearn.neighbors import KDTree

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from psbody.mesh import Mesh


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                "x.dim() is expected to be 2 or 3, but received {}".format(x.dim())
            )
        x = self.layer(x)
        return x

    def __repr__(self):
        return "{}({}, {}, seq_length={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.seq_length,
        )


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralEnblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform.to(x.device))
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super(SpiralDeblock, self).__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform.to(x.device))
        out = F.elu(self.conv(out))
        return out


def _next_ring(mesh, last_ring, other):
    res = []

    def is_new_vertex(idx):
        return idx not in last_ring and idx not in other and idx not in res

    for vh1 in last_ring:
        vh1 = om.VertexHandle(vh1)
        after_last_ring = False
        for vh2 in mesh.vv(vh1):
            if after_last_ring:
                if is_new_vertex(vh2.idx()):
                    res.append(vh2.idx())
            if vh2.idx() in last_ring:
                after_last_ring = True
        for vh2 in mesh.vv(vh1):
            if vh2.idx() in last_ring:
                break
            if is_new_vertex(vh2.idx()):
                res.append(vh2.idx())
    return res


def extract_spirals(mesh, seq_length, dilation=1):
    # output: spirals.size() = [N, seq_length]
    spirals = []
    for vh0 in mesh.vertices():
        reference_one_ring = []
        for vh1 in mesh.vv(vh0):
            reference_one_ring.append(vh1.idx())
        spiral = [vh0.idx()]
        one_ring = list(reference_one_ring)
        last_ring = one_ring
        next_ring = _next_ring(mesh, last_ring, spiral)
        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(mesh, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            kdt = KDTree(mesh.points(), metric="euclidean")
            spiral = kdt.query(
                np.expand_dims(mesh.points()[spiral[0]], axis=0),
                k=seq_length * dilation,
                return_distance=False,
            ).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(spiral[: seq_length * dilation][::dilation])
    return spirals


def makedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))


def to_sparse(spmat):
    return torch.sparse.FloatTensor(
        torch.LongTensor([spmat.tocoo().row, spmat.tocoo().col]),
        torch.FloatTensor(spmat.tocoo().data),
        torch.Size(spmat.tocoo().shape),
    )


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation)
    )
    return spirals


def spiral_tramsform(transform_fp, template_fp, ds_factors, seq_length, dilation):
    if not os.path.exists(transform_fp):
        print("Generating transform matrices...")
        mesh = Mesh(filename=template_fp)
        # ds_factors = [3.5, 3.5, 3.5, 3.5]
        _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(  # type: ignore
            mesh, ds_factors
        )
        tmp = {
            "vertices": V,
            "face": F,
            "adj": A,
            "down_transform": D,
            "up_transform": U,
        }

        with open(transform_fp, "wb") as fp:
            pickle.dump(tmp, fp)
        print("Done!")
        print("Transform matrices are saved in '{}'".format(transform_fp))
    else:
        with open(transform_fp, "rb") as f:
            tmp = pickle.load(f, encoding="latin1")

    spiral_indices_list = [
        preprocess_spiral(
            tmp["face"][idx], seq_length[idx], tmp["vertices"][idx], dilation[idx]
        )  # .to(device)
        for idx in range(len(tmp["face"]) - 1)
    ]

    down_transform_list = [
        to_sparse(down_transform)  # .to(device)
        for down_transform in tmp["down_transform"]
    ]
    up_transform_list = [
        to_sparse(up_transform) for up_transform in tmp["up_transform"]  # .to(device)
    ]

    return spiral_indices_list, down_transform_list, up_transform_list, tmp
