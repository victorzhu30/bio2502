binary_map = {
        'A': '00',
        'T': '01',
        'C': '10',
        'G': '11'
    }

reverse_binary_map = {v: k for k, v in binary_map.items()}

def binary_compress(dna_sequence):
    # with open(dna_sequence_file, 'r') as f:
    #     next(f)
    #     dna_sequence = ''.join([line.strip() for line in f])
    binary_str = ''.join([binary_map[c] for c in dna_sequence])

    total_bits = len(binary_str)
    padding_bits = (8 - (total_bits % 8)) % 8  
    binary_str += '0' * padding_bits

    bytes_list = []
    bytes_list.append(padding_bits.to_bytes(1, 'big'))

    for i in range(0, len(binary_str), 8):
        byte_str = binary_str[i:i+8]
        bytes_list.append(int(byte_str, 2).to_bytes(1, 'big'))

    return bytes_list

def save_bytes_list(bytes_list):
    with open('./output/compressed_sequence.bin', 'wb') as f:
        for b in bytes_list:
            f.write(b)

def binary_decompress(compressed_dna_sequence_file):
    with open(compressed_dna_sequence_file, 'rb') as f:
        bytes_data = f.read()
    
    padding_bit_length = bytes_data[0]

    binary_str = ''.join([format(byte, '08b') for byte in bytes_data])
 
    start_idx = 8 
    end_idx = len(binary_str) - padding_bit_length

    dna_restored = []   
    for i in range(start_idx, end_idx, 2):
        code = binary_str[i:i+2]
        dna_restored.append(reverse_binary_map[code])

    # print(''.join(dna_restored))  # 输出：ATCGATC
    with open('./output/decompressed_sequence.fasta', 'w') as f:
        f.write(''.join(dna_restored))

if __name__ == '__main__':
    bytes_list = binary_compress('./data/Ecoli.fasta')
    save_bytes_list(bytes_list)
    binary_decompress('./compressed_sequence.bin')