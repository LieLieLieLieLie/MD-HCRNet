"""
GraphHand — GCN on hand skeleton graph (SCI 1区 journal level).

Inspired by: Ge et al., "3D Hand Shape and Pose Estimation from a Single RGB
Image", CVPR 2019; and Zhao et al., "Semantic Graph Convolutional Networks
for 3D Human Pose Regression", CVPR 2019; plus 2021-2022 follow-up journal works
applying spectral GCN to hand mesh / pose estimation.

Core idea:
  1. ResNet-50 backbone → global average pool → initial joint estimates (21×3)
  2. Hand skeleton graph adjacency matrix (kinematic tree)
  3. 3-layer spectral GCN refines joint features using skeleton topology
  4. Final MLP head → refined 3D joint positions

Key distinction from direct regression:
  GCN enforces kinematic skeleton structure as inductive bias — joints share
  information with anatomically adjacent joints, improving articulated pose.

Model name for tables: "GraphHand"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ── MANO-21 skeleton adjacency (kinematic tree) ───────────────────────────────
#
# Joint indices (MANO-21):
#  0:Wrist  1-3:Index(MCP,PIP,DIP)  4-6:Middle  7-9:Pinky  10-12:Ring  13-15:Thumb
#  16:Index TIP  17:Middle TIP  18:Pinky TIP  19:Ring TIP  20:Thumb TIP
#
_MANO21_EDGES = [
    (0, 1),  (1, 2),  (2, 3),  (3, 16),   # Index chain
    (0, 4),  (4, 5),  (5, 6),  (6, 17),   # Middle chain
    (0, 7),  (7, 8),  (8, 9),  (9, 18),   # Pinky chain
    (0, 10), (10,11), (11,12), (12,19),   # Ring chain
    (0, 13), (13,14), (14,15), (15,20),   # Thumb chain
]

NUM_JOINTS = 21


def _build_adj(num_nodes: int, edges: list) -> torch.Tensor:
    """Build normalised adjacency D^{-1/2} A D^{-1/2} with self-loops."""
    A = torch.zeros(num_nodes, num_nodes)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(num_nodes)          # self-loops
    D_inv_sqrt = A.sum(-1).pow(-0.5).diag()
    return D_inv_sqrt @ A @ D_inv_sqrt    # (N, N)


class _GCNLayer(nn.Module):
    """Single spectral graph conv: H' = σ(A_norm · H · W)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.norm   = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, N, F_in)
            adj: (N, N) normalised adjacency (on same device)
        Returns:
            (B, N, F_out)
        """
        # Message passing: aggregate neighbours
        agg = torch.bmm(adj.unsqueeze(0).expand(x.shape[0], -1, -1), x)  # (B,N,F)
        out = self.norm(F.gelu(self.linear(agg)))
        return out


class GraphHand(nn.Module):
    """
    Graph Convolutional Network for 3D hand pose estimation.

    Input:  RGB image (B, 3, 256, 192)
    Output: joints3d (B, 21, 3) in camera space (metres)

    Model name: "GraphHand"
    """

    MODEL_NAME = "GraphHand"

    def __init__(self, num_joints: int = NUM_JOINTS,
                 gcn_hidden: int = 256, gcn_layers: int = 3):
        super().__init__()
        self.num_joints = num_joints

        # ── Adjacency matrix (persistent buffer) ─────────────────────────────
        self.register_buffer("adj", _build_adj(num_joints, _MANO21_EDGES))

        # ── ResNet-50 backbone ────────────────────────────────────────────────
        weights  = tvm.ResNet50_Weights.IMAGENET1K_V2
        backbone = tvm.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B,2048,1,1)

        feat_dim = 2048

        # ── Initial joint estimation (global feature → J×3) ──────────────────
        self.init_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_joints * 3),
        )
        nn.init.zeros_(self.init_head[-1].weight)
        nn.init.zeros_(self.init_head[-1].bias)

        # ── Per-joint feature projection (global → per-joint embedding) ───────
        self.joint_embed = nn.Linear(feat_dim, gcn_hidden)

        # ── GCN refinement layers ─────────────────────────────────────────────
        self.gcn = nn.ModuleList()
        in_ch = gcn_hidden + 3   # joint embedding + initial xyz estimate
        for i in range(gcn_layers):
            out_ch = gcn_hidden
            self.gcn.append(_GCNLayer(in_ch, out_ch))
            in_ch = out_ch

        # ── Residual output head ──────────────────────────────────────────────
        self.out_head = nn.Linear(gcn_hidden, 3)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, rgb: torch.Tensor,
                focal: torch.Tensor = None,
                cx: torch.Tensor = None,
                cy: torch.Tensor = None,
                depth: torch.Tensor = None) -> dict:
        B   = rgb.shape[0]
        dev = rgb.device
        J   = self.num_joints

        # ── Backbone ──────────────────────────────────────────────────────────
        feat = self.backbone(rgb).flatten(1)          # (B, 2048)

        # ── Initial joint estimates ────────────────────────────────────────────
        init_xyz = self.init_head(feat).view(B, J, 3)  # (B, J, 3)

        # ── Per-joint feature nodes: broadcast global feature to each joint ───
        joint_feat = self.joint_embed(feat)            # (B, gcn_hidden)
        node_feat  = joint_feat.unsqueeze(1).expand(-1, J, -1)  # (B, J, gcn_hidden)

        # ── Concatenate initial estimates as node features ────────────────────
        nodes = torch.cat([node_feat, init_xyz], dim=-1)   # (B, J, gcn_hidden+3)

        # ── GCN layers ────────────────────────────────────────────────────────
        for layer in self.gcn:
            nodes = layer(nodes, self.adj)

        # ── Residual refinement ───────────────────────────────────────────────
        delta    = self.out_head(nodes)                # (B, J, 3)
        joints3d = init_xyz + delta                    # (B, J, 3)

        dummy_2d = torch.zeros(B, J, 2, device=dev)
        return dict(
            joints3d    = joints3d,
            joints3d_s1 = init_xyz,   # before GCN = "stage 1" for ablation
            joints2d    = dummy_2d,
            joints2d_s1 = dummy_2d,
        )
