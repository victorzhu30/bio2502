import re
import pandas as pd

def preprocess(input_fasta_file):
    header_info = []
    header_pos = []
    non_ACGT_info = []
    non_ACGT_pos = []
    sequence = ''
    with open(input_fasta_file, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                line = line.upper().strip()
            sequence += line
    for i in range(len(sequence)):
        if sequence[i] == '>' or sequence[i] == '\n':
            header_pos.append(i)
    for i in range(0, len(header_pos), 2):
        header_info.append(sequence[header_pos[i]:header_pos[i+1]])
    """
    E.g.
    >NC_000913.3 Escherichia coli str. K-12 substr. MG1655, complete genome length = 71
    start_pos = 0
    end_pos = 71 = 70 + \n
    decompress时每个header_info末尾需要添加\n才能与start_pos和end_pos对应
    """
    sequence = re.sub(r'>.+?\n', '', sequence)
    i = 0
    while i < len(sequence):
        if sequence[i] not in 'ACGT':
            start = i
            end = i + 1
            while sequence[end] not in 'ACGT':
                end += 1
            i = end
            non_ACGT_info.append(sequence[start:end])
            non_ACGT_pos.append(start)
            non_ACGT_pos.append(end - 1)
        else:
            i += 1
    """
    E.g. ATCGANNCT 012345678
    start = 5
    end = 7
    i = 7
    """
    sequence = re.sub(r'[^ACGT]', '', sequence)
    return header_info, header_pos, non_ACGT_info, non_ACGT_pos, sequence
# decompress时先根据non_ACGT_info和non_ACGT_pos

if __name__ == '__main__':
    header_info, header_pos, non_ACGT_info, non_ACGT_pos, sequence = preprocess('./data/test.fasta')
    print(header_info)
    print(header_pos)
    print(non_ACGT_info)
    print(non_ACGT_pos)
    # print(sequence)