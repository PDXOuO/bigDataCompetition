import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LearnablePositionalEncoding(nn.Module):
    """可学习的绝对位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class MarketContextEncoder(nn.Module):
    """市场上下文编码器 - 学习市场整体状态作为条件"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.market_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, stock_features):
        market_feature = stock_features.mean(dim=1, keepdim=True)
        market_encoding = self.market_encoder(market_feature)
        gate = self.gate(torch.cat([stock_features, market_encoding.expand_as(stock_features)], dim=-1))
        return stock_features + gate * market_encoding


class TemporalCNNBlock(nn.Module):
    """提取局部微观形态的 1D-CNN 残差块"""

    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = x.transpose(1, 2)
        return self.activation(x + residual)


class MultiScaleTemporalBlock(nn.Module):
    """多尺度时间卷积，同时捕获不同时间尺度的形态特征"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        branch_dim = d_model // 3
        self.branch1 = nn.Sequential(
            nn.Conv1d(d_model, branch_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(d_model, branch_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(d_model, d_model - 2 * branch_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model - 2 * branch_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.branch_dim = branch_dim
        self.out_dim = d_model - 2 * branch_dim
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        b1 = self.branch1(x_t)
        b2 = self.branch2(x_t)
        b3 = self.branch3(x_t)
        merged = torch.cat([b1, b2, b3], dim=1)
        merged = merged.transpose(1, 2)
        return self.fusion(merged + x)


class TemporalAttentionPooling(nn.Module):
    """时间注意力池化 - 对60天序列加权聚合"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)
        return self.dropout(pooled)


class CrossStockAttention(nn.Module):
    """增强的股票间注意力 - 模拟市场内股票之间的相互作用"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, stock_features, attn_mask=None):
        batch_size, num_stocks, _ = stock_features.size()

        q = self.q_proj(stock_features)
        k = self.k_proj(stock_features)
        v = self.v_proj(stock_features)

        q = q.view(batch_size, num_stocks, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_stocks, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_stocks, self.nhead, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attended = torch.matmul(attn, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_stocks, self.d_model)

        output = self.norm(stock_features + self.dropout(self.out_proj(attended)))
        output = self.norm(output + self.dropout(self.ffn(output)))
        return output


class FeatureAttention(nn.Module):
    """特征注意力模块 - 使用门控机制"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_weights = torch.softmax(self.attention(x), dim=1)
        gate_values = self.gate(x)
        attended = torch.sum(x * attention_weights * gate_values, dim=1)
        return self.dropout(attended)


class StockTransformer(nn.Module):
    """股票排序Transformer - 结合多尺度时间特征和市场内股票交互"""

    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.model_type = 'RankingTransformer'
        self.config = config
        self.num_stocks = num_stocks

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.GELU(),
            nn.Dropout(config['dropout'])
        )

        self.pos_encoder = LearnablePositionalEncoding(
            config['d_model'], config['sequence_length'], config['dropout']
        )

        self.temporal_cnn = MultiScaleTemporalBlock(config['d_model'], config['dropout'])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True,
            activation='gelu'
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        self.market_context = MarketContextEncoder(config['d_model'], config['dropout'])
        self.temporal_pooling = TemporalAttentionPooling(config['d_model'], config['dropout'])
        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])
        self.cross_stock_attention = CrossStockAttention(config['d_model'], config['nhead'], config['dropout'])

        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.GELU(),
            nn.Dropout(config['dropout'] * 0.5)
        )

        self.score_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.GELU(),
            nn.Dropout(config['dropout'] * 0.3),
            nn.Linear(config['d_model'] // 4, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src, stock_adj_mask=None):
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)

        src_proj = self.input_proj(src_reshaped)
        src_proj = self.pos_encoder(src_proj)

        cnn_features = self.temporal_cnn(src_proj)

        temporal_features = self.temporal_encoder(cnn_features)

        temporal_pooled = self.temporal_pooling(temporal_features)

        attended_features = self.feature_attention(temporal_features)

        pooled_features = (temporal_pooled + attended_features) / 2

        stock_features = pooled_features.view(batch_size, num_stocks, -1)

        interactive_features = self.cross_stock_attention(stock_features, attn_mask=stock_adj_mask)
        interactive_features = self.market_context(interactive_features)

        interactive_features = interactive_features.view(batch_size * num_stocks, -1)

        ranking_features = self.ranking_layers(interactive_features)
        scores = self.score_head(ranking_features)

        output = scores.view(batch_size, num_stocks)
        return output
