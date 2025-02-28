import re

def extract_test_metrics_as_list(file_path):
    """
    从日志文件中提取 test 阶段的 f1、acc 和 kappa 指标，按 S0、S1、S2 顺序输出 9 个数。

    Args:
        file_path (str): 包含日志信息的 .log 文件路径。

    Returns:
        list: 包含 9 个数字的列表，顺序为 S0 的 f1, acc, kappa -> S1 的 f1, acc, kappa -> S2 的 f1, acc, kappa。
    """
    # 定义正则表达式匹配 test 阶段的 f1、acc 和 kappa
    pattern = re.compile(
        r'epoch=\d+, setting S([0-2]), test_acc=([-+]?\d*\.\d+|\d+), test_f1=([-+]?\d*\.\d+|\d+), test_kappa=([-+]?\d*\.\d+|\d+)'
    )
    
    # 用于存储提取的指标
    metrics = {0: {'f1': None, 'acc': None, 'kappa': None},
               1: {'f1': None, 'acc': None, 'kappa': None},
               2: {'f1': None, 'acc': None, 'kappa': None}}

    with open(file_path, 'r') as file:
        content = file.read()
        
        # 查找所有符合条件的行
        matches = pattern.findall(content)
        
        # 按最新的 epoch 更新 S0, S1, S2 的值
        for match in matches:
            setting = int(match[0])  # S0, S1, S2 的编号
            acc = float(match[1])
            f1 = float(match[2])
            kappa = float(match[3])
            
            # 更新指标到对应的测试集
            metrics[setting] = {'f1': f1, 'acc': acc, 'kappa': kappa}

    # 按 S0, S1, S2 顺序拼接成列表
    result = [
        metrics[0]['f1'], metrics[0]['acc'], metrics[0]['kappa'],
        metrics[1]['f1'], metrics[1]['acc'], metrics[1]['kappa'],
        metrics[2]['f1'], metrics[2]['acc'], metrics[2]['kappa']
    ]

    return result

# 输入日志文件路径
log_file_path = 'drugbank_01-19_finger_65/2025-01-19 09:07:55.log'  # 替换为实际的 .log 文件路径

# 提取 test 阶段指标
test_metrics = extract_test_metrics_as_list(log_file_path)

# 输出结果
print(", ".join(f"{value:.5f}" for value in test_metrics))
