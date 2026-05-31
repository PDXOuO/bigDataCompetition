# 配置参数
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sequence_length = 60
feature_num = '158+39'

config = {
    'sequence_length': sequence_length,
    'd_model': 192,
    'nhead': 6,
    'num_layers': 2,
    'dim_feedforward': 384,
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 3e-5,
    'dropout': 0.2,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    # 损失函数权重优化
    'pairwise_weight': 2.0,
    'base_weight': 1.0,
    'top5_weight': 15.0,
    'ic_weight': 2.0,

    # 训练策略
    'warmup_epochs': 5,
    'weight_decay': 2e-4,
    'swa_start_ratio': 0.7,

    'output_dir': os.path.join(project_root, 'model', f'{sequence_length}_{feature_num}'),
    'data_path': os.path.join(project_root, 'data'),
}
