import zipfile
from pathlib import Path

# ======================== 工具函数 ========================
def calculate_pcr(size_after: int, size_before: int) -> float:
    """
    计算通用压缩率（PCR）
    
    参数:
        size_after (int): 压缩后数据大小（单位：位）
        size_before (int): 压缩前数据大小（单位：位）
    
    返回:
        float: 压缩百分比，例如25.0表示压缩后为原大小的25%
    """
    if size_before == 0:
        raise ValueError("压缩前大小不能为0")
    return (size_after / size_before) * 100

def calculate_pcr_s(f: int, n: int) -> float:
    """
    计算子段压缩率（PCR_s）
    
    参数:
        f (int): 子段出现的频率（次数）
        n (int): 子段的长度（字符数）
    
    返回:
        float: PCR_s百分比
    """
    # 压缩后大小 = 存储子段本身(n*8位) + 存储f次位置差异(每次8位)
    compressed_size = f * 8 + n * 8
    # 原始大小 = 子段出现f次，每次占n*8位
    original_size = f * n * 8
    return calculate_pcr(compressed_size, original_size)

def calculate_pcr_b(f: int, n: int) -> float:
    """
    计算二进制压缩率（PCR_b）
    
    参数:
        f (int): 子段出现的频率（次数）
        n (int): 子段的长度（字符数）
    
    返回:
        float: PCR_b百分比（固定为25%）
    """
    # 压缩后大小 = 每个字符用2位存储，总大小f*n*2位
    compressed_size = f * n * 2
    # 原始大小 = 每个字符用8位存储，总大小f*n*8位
    original_size = f * n * 8
    return calculate_pcr(compressed_size, original_size)

def create_zip_archive(output_filename, files_to_compress):
    """
    创建 ZIP 压缩文件
    
    Args:
        output_filename: 输出的 ZIP 文件名（如 "archive.zip"）
        files_to_compress: 要压缩的文件列表（可以是字符串或 Path 对象）
    """
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_compress:
            # 转换为 Path 对象以便处理路径
            path = Path(file_path)
            
            if not path.exists():
                print(f"Warning: 文件 {path} 不存在，跳过压缩")
                continue
                
            # 在压缩文件中使用相对路径存储
            arcname = path.name  # 仅存储文件名（不包括目录路径）
            zipf.write(path, arcname)
            
    print(f"成功创建 {output_filename}，包含 {len(files_to_compress)} 个文件")