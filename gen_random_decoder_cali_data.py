import numpy as np
import os
import shutil
from tqdm import tqdm
from zipfile import ZipFile

# 设置随机种子（可选）
# np.random.seed(42)

def generate_init_target_seq():
    """生成init_target_seq数据"""
    return np.random.randint(low=1, high=91, size=(1, 26), dtype=np.int64)

def generate_out_enc():
    """生成out_enc数据"""
    return np.random.randn(1, 200, 512).astype(np.float32)

def generate_src_mask():
    """生成src_mask数据"""
    return np.ones((1, 200), dtype=np.float32)

def generate_step():
    """生成step数据"""
    return np.random.randint(low=1, high=21, size=(1,), dtype=np.int64)

def main():
    # 定义数据生成函数映射
    data_generators = {
        'init_target_seq': generate_init_target_seq,
        'out_enc': generate_out_enc,
        'src_mask': generate_src_mask,
        'step': generate_step
    }
    
    # 创建临时目录
    temp_dir = 'temp_data'
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 为每种数据类型生成一组，每组32个
        for tensor_name, generator in data_generators.items():
            print(f"\n生成 {tensor_name} 数据:")
            
            group_dir = os.path.join(temp_dir, tensor_name)
            os.makedirs(group_dir, exist_ok=True)
            
            # 生成32个数据
            for data_idx in tqdm(range(1, 33)):
                # 生成数据
                data = generator()
                
                # 保存为.npy文件
                file_path = os.path.join(group_dir, f'{tensor_name}_{data_idx}.npy')
                np.save(file_path, data)
            
            # 压缩文件夹
            zip_path = f'cali_data/decoder_data/{tensor_name}'
            with ZipFile(f'{zip_path}.zip', 'w') as zipf:
                for root, _, files in os.walk(group_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
            
            print(f"  {tensor_name} 数据已压缩为 {zip_path}.zip")
    
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()