import re

def extract_test_metrics(file_path):
    """
    从日志文件中仅提取测试集上的 prc_auc、roc_auc 和 ap 指标，按 S0、S1、S2 的顺序输出。

    Args:
        file_path (str): 日志文件路径。

    Returns:
        list: 包含 9 个数字的列表，顺序为 S0 的 prc_auc, roc_auc, ap -> S1 的 prc_auc, roc_auc, ap -> S2 的 prc_auc, roc_auc, ap。
    """
    # 定义正则表达式匹配测试集上的指标
    pattern = re.compile(
        r'test best_checkpoint_path=.*?setting S([0-2]), roc_auc=([-+]?\d*\.\d+|\d+),prc_auc=([-+]?\d*\.\d+|\d+),ap=([-+]?\d*\.\d+|\d+)'
    )
    
    # 用于存储每组的指标
    metrics = {0: {'prc_auc': None, 'roc_auc': None, 'ap': None},
               1: {'prc_auc': None, 'roc_auc': None, 'ap': None},
               2: {'prc_auc': None, 'roc_auc': None, 'ap': None}}

    with open(file_path, 'r') as file:
        content = file.read()
        
        # 查找所有匹配的行
        matches = pattern.findall(content)
        
        # 更新每个设置的指标
        for match in matches:
            setting = int(match[0])  # S0, S1, S2 的编号
            roc_auc = float(match[1])
            prc_auc = float(match[2])
            ap = float(match[3])
            
            # 存储提取的指标
            metrics[setting] = {'prc_auc': prc_auc, 'roc_auc': roc_auc, 'ap': ap}

    # 按 S0, S1, S2 顺序拼接结果
    result = [
        metrics[0]['prc_auc'], metrics[0]['roc_auc'], metrics[0]['ap'],
        metrics[1]['prc_auc'], metrics[1]['roc_auc'], metrics[1]['ap'],
        metrics[2]['prc_auc'], metrics[2]['roc_auc'], metrics[2]['ap']
    ]

    return result

# 输入日志文件路径
log_file_path = 'twosides_01-20_finger_60/2025-01-20 20:56:06.log'  # 替换为实际的 .log 文件路径

# 提取指标
metrics = extract_test_metrics(log_file_path)

# 输出结果
print(", ".join(f"{value:.4f}" for value in metrics))
