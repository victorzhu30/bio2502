from preprocess_fasta import preprocess
from util import calculate_pcr_s, calculate_pcr_b, create_zip_archive
from binary_compression import binary_compress, save_bytes_list
import pandas as pd
from collections import defaultdict
from datetime import datetime
import sys
import pickle

class SeqCompress:
    def __init__(self, n=8, m=6):
        self._n = n             # 子段长度
        self._m = m             # 最多选择m个高频子段
        self._segments = defaultdict(int)    # 存储子段及其频率
        self._best_segments = []
        self._result = {}

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if not isinstance(value, int):
            raise ValueError("n must be int")
        if value <= 0:
            raise ValueError("n must be positive")
        self._n = value

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, value):
        if not isinstance(value, int):
            raise ValueError("m must be int")
        if value <= 0:
            raise ValueError("m must be positive")
        self._m = value

    @property
    def segments(self):
        return self._segments

    @property
    def best_segments(self):
        return self._best_segments

    @property
    def result(self):
        return self._result

    def find_frequent_segments(self, sequence: str) -> None:
        """
        检测高频子段（滑动窗口法）
        
        参数:
            sequence (str): 预处理后的纯ACGT序列
        """
        self._segments.clear()
        # 滑动窗口遍历序列，步长为1
        # E.g. sequence = 'ATCGATC' self._n = 3 len(sequence) - self._n + 1 = 5 i = 0, 1, 2, 3, 4
        # segment = sequence[i:i+self._n] = sequence[0:3], sequence[1:4], sequence[2:5], sequence[3:6], sequence[4:7]
        for i in range(len(sequence) - self._n + 1):
            segment = sequence[i:i+self._n]
            # 统计频率
            # self._segments = {}
            # if segment in self._segments:
            #     self._segments[segment] += 1
            # else:
            #     self._segments[segment] = 1
            self._segments[segment] += 1
            # defaultdict 自动初始化键值，避免了 if-else 判断，代码更简洁。

    def select_best_segments(self) -> list:
        """
        选择最优压缩子段（PCR_s < PCR_b）
        
        返回:
            list: 返回保留的子段列表，格式[(segment, frequency)]
        """
        self._best_segments.clear()
        # 按频率从高到低排序
        sorted_segments = sorted(self._segments.items(), key=lambda x: x[1], reverse=True)
        
        for segment, freq in sorted_segments:
            # 计算PCR_s和PCR_b
            pcr_s = calculate_pcr_s(freq, self._n)
            pcr_b = calculate_pcr_b(freq, self._n)
            
            # 决策：仅保留PCR_s更优的子段，且不超过m个
            if pcr_s < pcr_b and len(self._best_segments) < self._m:
                self._best_segments.append((segment, freq))
            else:
                break  # 后续频率更低，无需检查

    def compress_sequence(self, sequence: str) -> dict:
        """
        执行完整压缩流程
        
        参数:
            sequence (str): 预处理后的纯ACGT序列
            
        返回:
            dict: 压缩结果，包含子段信息、剩余序列二进制编码等
        """
        # 1. 检测高频子段
        self.find_frequent_segments(sequence)
        # 2. 选择最优子段
        self.select_best_segments()
        # 3. 移除子段
        self.remove_high_frequency_segments(original_sequence=sequence)
        # 4.编码剩余部分
        self.encode_remaining(remaining_sequence=self._result['remaining_seq'])

    def remove_high_frequency_segments(self, original_sequence:str) -> str:
        """
        模型搜索长度为n的最频繁的重复子分段。
        子段将作为子段（一次）和子段位置在输入序列中的增量差异保存到单独的文件中。
        随后，将从 input 序列中删除子区段。

        问题：
        find_frequent_segments在统计时会出现overlap（滑动窗口）
        删除A子区段可能会影响B子区段，导致B子区段消失
            
        返回:
            dict: 包含压缩后的各部分数据
        """
        # 初始化结果字典
        result = {
            'segments': [],      # 存储子段及其位置差
            'remaining_seq': original_sequence  # 初始为原序列，逐步删除子段
        }
        
        # 处理每个保留的子段
        for seg, _ in self._best_segments:
            positions = []
            start = 0
            # 查找所有出现位置
            # start作为指针，从0开始，逐步向后移动
            # 没找到，start+1
            # 找到，position.append(start)，start+=n
            """
            ATCGGATCCATCCTT len = 15
            ATC n = 3
            start = 0 <= 15 - 3 = 12
            position = [0]
            start = 3 <= 12
            start = 4 <= 12
            start = 5 <= 12
            positions = [0, 5]
            start = 8 <= 12
            start = 9 <= 12
            positions = [0, 5, 9]
            start = 12 <= 12
            start = 13 > 12

            postions = [0, 5, 9]

            """
            while start <= len(result['remaining_seq']) - self._n:
                if result['remaining_seq'][start:start + self._n] == seg:
                    positions.append(start)
                    start += self._n  # 跳过已处理部分
                else:
                    start += 1
            
            # 存储子段信息：子段内容、位置差
            if positions:
                # 计算位置差
                deltas = [positions[0]]  # 第一个绝对位置
                for i in range(1, len(positions)):
                    deltas.append(positions[i] - positions[i-1])
                """
                deltas = [0, 5, 4]
                """
                result['segments'].append({
                    'segment': seg,
                    'deltas': deltas,
                })
                
                # 从剩余序列中删除该子段的所有出现
                result['remaining_seq'] = result['remaining_seq'].replace(seg, '')
                """
                ATCGGATCCATCCTT -> GGCCTT

                复原：
                GGCCTT
                0: ATCGGCCTT
                0+5:ATCGGATCCCTT
                0+5+4:ATCGGATCCATCCTT
                """
        self._result = result
        # decompress需要倒序组装，即先添加最后一个子段的所有出现位置

    def encode_remaining(self, remaining_sequence: str):
        """
        对剩余序列进行二进制编码
        """
        bytes_list = binary_compress(remaining_sequence)
        save_bytes_list(bytes_list)

if __name__ == "__main__":
    args = sys.argv  # 获取所有参数
    if len(args) != 2 and len(args) != 4 :
        print("Usage: python3 SeqCompress.py <input_fasta_file> [Optional: n m]")
        print("n - Length of segments (default: 8)")
        print("m - Maximum number of segments to select (default: 6)")
        sys.exit(1)

    fasta_file_path = args[1]  # 获取输入的FASTA文件路径
    if not fasta_file_path.endswith('.fasta'):
        print("Error: Please provide a valid FASTA file with .fasta extension.")
        sys.exit(1)

    starttime = datetime.now()
    # 调用预处理函数
    header_info, header_pos, non_ACGT_info, non_ACGT_pos, sequence = preprocess(fasta_file_path)
    header_df = pd.DataFrame([
        {
            "header": header_info[i], 
            "start": header_pos[i*2], 
            "end": header_pos[i*2+1]
        } 
        for i in range(len(header_info))
    ])
    non_ACGT_df = pd.DataFrame([
        {
            "non_ACGT": non_ACGT_info[i], 
            "start": non_ACGT_pos[i*2],
            "end": non_ACGT_pos[i*2+1]
        } 
        for i in range(len(non_ACGT_info))
    ])
    header_df.to_csv('./output/header.csv', index=False)
    non_ACGT_df.to_csv('./output/non_ACGT.csv', index=False)

    if len(args) == 4:
        try:
            n = int(args[2])
            m = int(args[3])
            if n <= 0 or m <= 0:
                raise ValueError("n and m must be positive integers")
            compressor = SeqCompress(n, m)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        compressor = SeqCompress()

    # 调用压缩流程
    compressor.compress_sequence(sequence)

    # 输出结果
    # print("Segments:", compressor.best_segments)
    # print("Compressed Segments:", compressor.result['segments'])
    # print("Remaining Sequence:", compressor.result['remaining_seq'])
    print("Compressed Data saved to ./output/compressed_sequence.bin")
    file_path_pickle = './output/segments_location.pkl'
    with open(file_path_pickle, 'wb') as f: # 'wb' 表示二进制写入
        pickle.dump(compressor.result['segments'], f)
    print(f"高频子段位置数据已存储到 {file_path_pickle}")

    files = [
        './output/header.csv',
        './output/non_ACGT.csv',
        './output/compressed_sequence.bin',
        './output/segments_location.pkl'
    ]
    create_zip_archive('./output/results.zip', files)
    print("所有生成的结果文件已使用gzip压缩到./output/results.zip")
    endtime = datetime.now()
    print(f"Total time taken: {endtime - starttime}")