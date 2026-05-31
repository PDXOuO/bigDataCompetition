import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized
import os
import json
import multiprocessing as mp
import random
from torch.optim.swa_utils import AveragedModel, SWALR


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


feature_cloums_map = {
    '39': ['instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
           'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5',
           'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60',
           'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10', 'high_low_spread',
           'open_close_spread', 'high_close_spread', 'low_close_spread'],
    '158+39': ['instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
           'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
           'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
           'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
           'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
           'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
           'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
           'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
           'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
           'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
           'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
           'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
           'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5',
           'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10',
           'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10',
           'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60', 'sma_5', 'sma_20',
           'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20',
           'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10',
           'volatility_20', 'return_1', 'return_5', 'return_10', 'high_low_spread', 'open_close_spread',
           'high_close_spread', 'low_close_spread']
}

feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}


def _build_label_and_clean(processed, drop_small_open=True):
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}数据为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="测试数据处理", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="训练数据处理", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    return _preprocess_common(df, stockid2idx, desc="验证数据特征工程", drop_small_open=True)


class ListMLELoss(nn.Module):
    """ListMLE 损失 - 专门为排序设计的损失函数"""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, y_pred, y_true):
        batch_size, num_items = y_true.size()
        device = y_pred.device

        sorted_indices = torch.argsort(y_true, dim=1, descending=True)
        y_pred_sorted = torch.gather(y_pred, 1, sorted_indices)

        listmle = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            valid_items = num_items
            for j in range(valid_items - 1):
                denom = torch.logsumexp(y_pred_sorted[i, j:] / self.temperature, dim=0)
                listmle[i] += y_pred_sorted[i, j] / self.temperature - denom

        return listmle.mean()


class LambdaRankLoss(nn.Module):
    """LambdaRank 损失 - 结合 NDCG 梯度的排序损失"""

    def __init__(self, k=5, ndcg_weight=2.0, listwise_weight=1.0):
        super().__init__()
        self.k = k
        self.ndcg_weight = ndcg_weight
        self.listwise_weight = listwise_weight

    def dcg_at_k(self, rel, k=None):
        if k is None:
            k = len(rel)
        rel = rel[:k]
        discounts = torch.log2(torch.arange(2, len(rel) + 2, device=rel.device, dtype=torch.float32))
        return torch.sum(rel / discounts)

    def ndcg_at_k(self, y_pred, y_true, k=None):
        if k is None:
            k = len(y_pred)
        k = min(k, len(y_pred))

        _, sorted_indices = torch.sort(y_pred, descending=True)
        sorted_rel = y_true[sorted_indices[:k]]

        ideal, _ = torch.sort(y_true, descending=True)
        ideal_rel = ideal[:k]

        dcg = self.dcg_at_k(sorted_rel, k)
        idcg = self.dcg_at_k(ideal_rel, k)

        return dcg / (idcg + 1e-8)

    def listwise_loss(self, y_pred, y_true):
        log_probs = F.log_softmax(y_pred, dim=1)
        normalized_true = (y_true - y_true.mean(dim=1, keepdim=True)) / (y_true.std(dim=1, keepdim=True) + 1e-8)
        normalized_true = torch.sigmoid(normalized_true)
        loss = -(normalized_true * log_probs).sum(dim=1)
        return loss.mean()

    def forward(self, y_pred, y_true):
        ndcg_loss = 0.0
        batch_size = y_pred.size(0)
        for i in range(batch_size):
            ndcg = self.ndcg_at_k(y_pred[i], y_true[i], k=self.k)
            ndcg_loss += (1.0 - ndcg)

        ndcg_loss = ndcg_loss / batch_size
        listwise = self.listwise_weight * self.listwise_loss(y_pred, y_true)

        return self.ndcg_weight * ndcg_loss + listwise


class CombinedRankingLoss(nn.Module):
    """组合排序损失 - ListMLE + LambdaRank + IC + Pairwise"""

    def __init__(self, temperature=1.0, k=5,
                 listmle_weight=1.0, lambdarank_weight=2.0,
                 pairwise_weight=1.0, ic_weight=3.0):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.listmle_weight = listmle_weight
        self.lambdarank_weight = lambdarank_weight
        self.pairwise_weight = pairwise_weight
        self.ic_weight = ic_weight

        self.listmle = ListMLELoss(temperature=temperature)
        self.lambdarank = LambdaRankLoss(k=k, ndcg_weight=1.0, listwise_weight=0.5)

    def ic_loss(self, y_pred, y_true, eps=1e-8):
        pred_mean = y_pred.mean(dim=1, keepdim=True)
        true_mean = y_true.mean(dim=1, keepdim=True)
        pred_centered = y_pred - pred_mean
        true_centered = y_true - true_mean
        cov = (pred_centered * true_centered).sum(dim=1)
        pred_var = torch.sqrt((pred_centered ** 2).sum(dim=1) + eps)
        true_var = torch.sqrt((true_centered ** 2).sum(dim=1) + eps)
        ic = cov / (pred_var * true_var)
        return 1.0 - ic.mean()

    def pairwise_loss(self, y_pred, y_true):
        batch_size, num_items = y_pred.size()
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        mask = (true_diff != 0).float()
        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (pairwise_loss * mask).sum(dim=[1, 2]) / num_pairs
        return loss.mean()

    def forward(self, y_pred, y_true):
        listmle_loss = self.listmle(y_pred, y_true)
        lambdarank_loss = self.lambdarank(y_pred, y_true)
        ic_loss = self.ic_loss(y_pred, y_true)
        pairwise = self.pairwise_loss(y_pred, y_true)

        total = (self.listmle_weight * listmle_loss +
                 self.lambdarank_weight * lambdarank_loss +
                 self.ic_weight * ic_loss +
                 self.pairwise_weight * pairwise)
        return total


class ImprovedRankingLoss(nn.Module):
    """改进的排序损失函数，结合 Listwise、Pairwise、LambdaRank 和 NDCG"""

    def __init__(self, temperature=0.5, k=5, weight_factor=15.0,
                 pairwise_weight=2.0, base_weight=1.0, ic_weight=2.0, ndcg_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight
        self.ic_weight = ic_weight
        self.ndcg_weight = ndcg_weight

    def dcg_at_k(self, rel, k=None):
        if k is None:
            k = len(rel)
        rel = rel[:k]
        discounts = torch.log2(torch.arange(2, len(rel) + 2, device=rel.device, dtype=torch.float32))
        return torch.sum(rel / discounts)

    def ndcg_at_k(self, pred_scores, true_relevance, k=None):
        if k is None:
            k = len(pred_scores)

        _, sorted_indices = torch.sort(pred_scores, descending=True)
        sorted_rel = true_relevance[sorted_indices]

        ideal, _ = torch.sort(true_relevance, descending=True)
        dcg = self.dcg_at_k(sorted_rel, k)
        idcg = self.dcg_at_k(ideal, k)

        ndcg = dcg / (idcg + 1e-8)
        return ndcg

    def listwise_loss(self, y_pred, y_true, weights):
        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)
        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        ce_loss = (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()
        return ce_loss

    def pairwise_loss(self, y_pred, y_true, weights):
        batch_size, num_items = y_pred.size()
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        mask = (true_diff != 0).float()
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        weighted_loss = pairwise_loss * mask * weight_matrix
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()
        return loss

    def ic_loss(self, y_pred, y_true, eps=1e-8):
        pred_mean = y_pred.mean(dim=1, keepdim=True)
        true_mean = y_true.mean(dim=1, keepdim=True)
        pred_centered = y_pred - pred_mean
        true_centered = y_true - true_mean

        cov = (pred_centered * true_centered).sum(dim=1)
        pred_var = torch.sqrt((pred_centered ** 2).sum(dim=1) + eps)
        true_var = torch.sqrt((true_centered ** 2).sum(dim=1) + eps)

        ic = cov / (pred_var * true_var)
        return 1.0 - ic.mean()

    def ndcg_loss(self, y_pred, y_true):
        batch_size = y_pred.size(0)
        total_ndcg_loss = 0.0
        for i in range(batch_size):
            ndcg = self.ndcg_at_k(y_pred[i], y_true[i], k=self.k)
            total_ndcg_loss += (1.0 - ndcg)
        return total_ndcg_loss / batch_size

    def forward(self, y_pred, y_true):
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)

        _, top_indices = torch.topk(y_true, k, dim=1)

        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor

        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        ic = self.ic_loss(y_pred, y_true)
        ndcg = self.ndcg_loss(y_pred, y_true)

        total_loss = listwise + (self.pairwise_weight * pairwise) + (self.ic_weight * ic) + (self.ndcg_weight * ndcg)
        return total_loss


def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    batch_size = y_pred.size(0)
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()

        if valid_indices.numel() < k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]

        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()

        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()

        random_return_sum = k * valid_true.mean().item()

        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0

        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)

    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
    }
    metrics['ratio_pred'] = np.mean(ratio_pred_list) if ratio_pred_list else 0.0
    metrics['ratio_random'] = np.mean(ratio_random_list) if ratio_random_list else 0.0
    metrics['final_score'] = np.mean(final_score_list) if final_score_list else 0.0

    return metrics


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),
            'targets': torch.FloatTensor(self.targets[idx]),
            'relevance': torch.LongTensor(self.relevance_scores[idx]),
            'stock_indices': torch.LongTensor(self.stock_indices[idx])
        }


def collate_fn(batch):
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]

    max_stocks = max(seq.size(0) for seq in sequences)

    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []

    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)

        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)

            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)

        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)

    return {
        'sequences': torch.stack(padded_sequences),
        'targets': torch.stack(padded_targets),
        'relevance': torch.stack(padded_relevance),
        'stock_indices': torch.stack(padded_stock_indices),
        'masks': torch.stack(masks)
    }


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, num_epochs, min_lr_ratio=0.1):
    """带 Warmup 的余弦退火学习率调度器"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer, scheduler=None):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        relevance = batch['relevance'].to(device)
        masks = batch['masks'].to(device)

        optimizer.zero_grad()

        outputs = model(sequences)

        masked_outputs = outputs * masks + (1 - masks) * (-1e9)
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks

        batch_loss = None
        batch_size = sequences.size(0)

        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()

            if valid_indices.numel() == 0:
                continue

            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)

            valid_pred = masked_outputs[i][valid_indices]
            valid_relevance = masked_relevance[i][valid_indices]

            if len(valid_pred) > 1:
                loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss

        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 5.0))
            if writer:
                writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch * len(dataloader) + local_step)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += batch_loss.item()

            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v

            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch * len(dataloader) + local_step)
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('train/learning_rate', current_lr, global_step=epoch * len(dataloader) + local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch * len(dataloader) + local_step)

    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step

    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics


def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch + 1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)

            outputs = model(sequences)

            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            batch_loss = None
            batch_size = sequences.size(0)

            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()

                if valid_indices.numel() == 0:
                    continue

                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)

                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]

                if len(valid_pred) > 1:
                    _, sorted_indices = torch.sort(valid_true, descending=True)

                    relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                    relevance_scores[sorted_indices] = torch.arange(len(valid_true), 0, -1, device=device,
                                                                      dtype=torch.float32)
                    relevance_scores = relevance_scores.detach()
                    loss = criterion(valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss

            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()

            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v

            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches

    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)

    return avg_loss, total_metrics


def split_train_val_by_last_month(df, sequence_length):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后一个月): {val_start.date()} 到 {last_date.date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


def cross_sectional_zscore(df, cols):
    """每日截面标准化，防止时间序列数据泄露和极端值影响"""
    print("正在进行截面 Z-Score 标准化...")
    mean = df.groupby('日期')[cols].transform('mean')
    std = df.groupby('日期')[cols].transform('std') + 1e-8
    df[cols] = (df[cols] - mean) / std
    return df


def main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    data_path = config['data_path']
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    data_file = os.path.join(data_path, 'train.csv')
    full_df = pd.read_csv(data_file)
    train_df, val_df, val_start = split_train_val_by_last_month(full_df, config['sequence_length'])

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)

    train_data[features] = train_data[features].fillna(0)
    val_data[features] = val_data[features].fillna(0)

    train_data = cross_sectional_zscore(train_data, features)
    val_data = cross_sectional_zscore(val_data, features)

    train_data[features] = train_data[features].clip(-5.0, 5.0)
    val_data[features] = val_data[features].clip(-5.0, 5.0)

    train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
        train_data, features, config['sequence_length'], ranking_data_path=config.get('train_ranking_data_path')
    )
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        val_data, features, config['sequence_length'], ranking_data_path=config.get('val_ranking_data_path'),
        min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"验证集样本数: {len(val_sequences)}")

    train_dataset = RankingDataset(
        train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = ImprovedRankingLoss(
        k=5, temperature=0.5, weight_factor=config.get('top5_weight', 15.0),
        pairwise_weight=config.get('pairwise_weight', 2.0), base_weight=config.get('base_weight', 1.0),
        ic_weight=config.get('ic_weight', 2.0), ndcg_weight=1.0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 5e-5)
    )

    warmup_epochs = config.get('warmup_epochs', 5)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, config['num_epochs'])
    swa_start = int(config['num_epochs'] * config.get('swa_start_ratio', 0.7))
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-6)

    if is_train:
        best_score = -float('inf')
        best_epoch = -1

        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch + 1}/{config['num_epochs']} ===")

            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer,
                scheduler=None if epoch >= warmup_epochs else scheduler
            )

            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")

            if epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                if writer:
                    writer.add_scalar('train/learning_rate', swa_scheduler.get_last_lr()[0], epoch)
                eval_model = swa_model.module
            else:
                if epoch < warmup_epochs:
                    for _ in range(len(train_loader)):
                        scheduler.step()
                if writer:
                    writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], epoch)
                eval_model = model

            eval_loss, eval_metrics = evaluate_ranking_model(
                eval_model, val_loader, criterion, device, writer, epoch
            )

            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")

            current_final_score = eval_metrics.get('final_score', 0.0)
            if current_final_score > best_score:
                best_score = current_final_score
                best_epoch = epoch + 1
                torch.save(eval_model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"保存最佳模型 - final score: {best_score:.4f} (SWA: {epoch >= swa_start})")

        print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 final score: {best_score:.4f}")
        with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
            f.write(f"Best epoch: {best_epoch}\nBest final_score: {best_score:.6f}\n")

        if writer:
            writer.close()

        return best_score


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 final score: {best_score:.4f} ##########")
