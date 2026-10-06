"""
Step 2 공용 모델 정의 (학습 스크립트 / 추론 스크립트가 함께 import).

구조: Compound -> Protein 단방향 Cross-Attention
    compound tokens (Lc,768) --Linear+LN--> (Lc,d)   Query
    protein  tokens (Lp,1280) --Linear+LN--> (Lp,d)  Key / Value  (갱신되지 않고 그대로 참조됨)

    N x [ Pre-LN Cross-Attention (Q = compound, K = V = protein)  ->  FFN ]
        -> compound 토큰만 갱신된다 ("각 화합물 토큰이 단백질의 어느 residue 를 보는가")
        -> protein 쪽으로 되돌아가는 방향(protein query -> compound key/value)은 없다

    compound 토큰 masked mean + max pooling -> shared body -> task heads (paffinity 회귀)

    단백질 정보는 오직 cross-attention 의 key/value 를 통해서만 예측에 반영되므로,
    attention weight (화합물 토큰 x 단백질 residue) 가 곧 "모델이 단백질의 어디를 봤는가"이다.

forward(..., return_attn=True) 로 층별 attention 을 꺼낼 수 있다.
    info["attn_c2p"] : 층별 (B, Lc, Lp)  (head 평균, 단백질 padding 위치는 0)
"""

import torch
import torch.nn as nn


COMP_DIM = 768
PROT_DIM = 1280

# 체크포인트 호환성 확인용 (양방향 구조로 학습된 예전 체크포인트와 구분)
ARCHITECTURE = "c2p_v1"


class CrossAttentionBlock(nn.Module):
    """query 시퀀스가 context(key/value) 시퀀스를 attend (Pre-LN) + FFN."""

    def __init__(self, d_model, n_heads, ffn_dim, dropout):

        super().__init__()

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.drop = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model)
        )

    def forward(self, query, context, context_valid, return_attn=False):

        q = self.norm_q(query)
        kv = self.norm_kv(context)

        # return_attn=True 이면 head 평균된 attention (B, Lq, Lk) 반환
        out, attn = self.attn(
            q, kv, kv,
            key_padding_mask=~context_valid,     # True = 무시할 위치
            need_weights=return_attn,
            average_attn_weights=True
        )

        query = query + self.drop(out)
        query = query + self.drop(self.ffn(self.norm_ffn(query)))

        return query, attn


def masked_mean(x, valid):
    v = valid.unsqueeze(-1).to(x.dtype)
    return (x * v).sum(dim=1) / v.sum(dim=1).clamp(min=1.0)


def masked_max(x, valid):
    fill = torch.finfo(x.dtype).min
    return x.masked_fill(~valid.unsqueeze(-1), fill).max(dim=1).values


class CrossAttentionDTIRegressor(nn.Module):

    def __init__(
        self,
        task_names=("ic50",),
        comp_dim=COMP_DIM,
        prot_dim=PROT_DIM,
        d_model=256,
        n_heads=8,
        n_layers=2,
        ffn_dim=512,
        attn_dropout=0.1,
        body_dropout=0.3,
        architecture=ARCHITECTURE
    ):

        super().__init__()

        if architecture != ARCHITECTURE:
            raise ValueError(
                f"지원하지 않는 구조입니다: {architecture} (기대: {ARCHITECTURE})"
            )

        self.task_names = list(task_names)

        self.comp_proj = nn.Sequential(
            nn.Linear(comp_dim, d_model), nn.LayerNorm(d_model)
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, d_model), nn.LayerNorm(d_model)
        )

        # 각 층: Q = compound(이전 층 출력), K/V = protein(투영된 값 그대로)
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, ffn_dim, attn_dropout)
            for _ in range(n_layers)
        ])

        pooled_dim = d_model * 2   # compound mean + max

        # Step1과 동일한 shared body 구조
        self.shared_body = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(body_dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(body_dropout)
        )

        # Task heads (paffinity 회귀) - task_names 순서대로 출력 column 이 정해짐
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in self.task_names])

    def forward(self, comp, comp_valid, prot, prot_valid, return_attn=False):
        """
        return: preds (B, n_tasks)
                return_attn=True 이면 (preds, info)
                info = {"attn_c2p": [층별 (B, Lc, Lp)]}
        """

        c = self.comp_proj(comp.float())      # Query 쪽 (갱신됨)
        p = self.prot_proj(prot.float())      # Key/Value 쪽 (갱신되지 않음)

        attn_c2p = []

        for layer in self.layers:
            c, attn = layer(c, p, prot_valid, return_attn)
            if return_attn:
                attn_c2p.append(attn)

        pooled = torch.cat(
            [masked_mean(c, comp_valid), masked_max(c, comp_valid)],
            dim=1
        )

        features = self.shared_body(pooled)

        preds = torch.cat([head(features) for head in self.heads], dim=1)

        if return_attn:
            return preds, {"attn_c2p": attn_c2p}

        return preds

    def get_shared_layer(self):
        # GradNorm(task 2개 이상일 때)에서 gradient norm을 측정할 shared parameter
        return self.shared_body[4].weight
