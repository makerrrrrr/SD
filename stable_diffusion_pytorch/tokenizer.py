import unicodedata
import functools
import itertools
import json
from typing import List, Tuple, Text
import regex as re
from . import util


# 将字节值（0-255）映射到字符，将特殊字符（不可显示字符‘C’,分割符‘Z’）映射到新的范围，普通字符保留映射关系
def create_bytes_table() -> dict:
    table = {}
    special_count = 0
    for byte in range(256):
        category = unicodedata.category(chr(byte))
        # C 表示其他类别字符，通常为不可显示字符
        # Z 表示分割符，用于分割文本或者局部的作用
        if category[0] not in ['C', 'Z']:      # ith character is NOT control char or space
            table[byte] = chr(byte)
        else:                                  # ith character IS control char or space
            table[byte] = chr(special_count + 256)
            special_count += 1
    return table

def pairwise(seq):
    '''
    seq = ['p', 'h', 'o', 't', 'o', 'g', 'r', 'a', 'p', 'h</w>']
    a=['p', 'h', 'o', 't', 'o', 'g', 'r', 'a', 'p', 'h</w>']
    b=['h', 'o', 't', 'o', 'g', 'r', 'a', 'p', 'h</w>']
    return:
        [('p', 'h'), ('h', 'o'), ('o', 't'), 
        ('t', 'o'), ('o', 'g'), ('g', 'r'), 
        ('r', 'a'), ('a', 'p'), ('p', 'h</w>')]
    '''
    a = iter(seq)
    b = iter(seq)
    next(b)    #next(b) 指向‘h’
    return zip(a, b)

class Tokenizer:
    def __init__(self, ):
        with open(util.get_file_path('vocab.json'), encoding='utf-8') as f:
            self.vocab = json.load(f)

        with open(util.get_file_path('merges.txt'), encoding='utf-8') as f:
            lines = f.read().split('\n')

            lines = lines[1:-1]
            self.merges = {tuple(bigram.split()): i for i, bigram in enumerate(lines)}

        self.bos_token = self.vocab["<|startoftext|>"]
        self.eos_token = self.vocab["<|endoftext|>"]
        self.pad_token = self.vocab["<|endoftext|>"]
        self.max_length = 77
        self.bytes_table = create_bytes_table()
        self.chunk_pattern = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    #文本预处理
    def encode(self, text: str) -> List[int]:
        # 将文本标准化为NFC格式
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = text.lower()
        tokens = [self.bos_token]
        for chunk in re.findall(self.chunk_pattern, text): 
            chunk = ''.join(self.bytes_table[byte] for byte in chunk.encode('utf-8'))
            tokens.extend(self.vocab[word] for word in self.bpe(chunk))
            tokens.append(self.eos_token)
            # import pdb;pdb.set_trace()
            # for word in self.bpe(chunk):
            #     tokens.extend(self.vocab[word])
            # tokens.append(self.eos_token)

        tokens = tokens[:self.max_length]
        token_length = len(tokens)
        pad_length = self.max_length - token_length
        tokens += [self.pad_token] * pad_length
        return tokens
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(text) for text in texts] 
        # ls_tokens=[]
        # for text in texts:
        #     tokens=self.encode(text)
        #     ls_tokens.append(tokens)
        # return ls_tokens

    @functools.lru_cache(maxsize=10000)
    def bpe(self, chunk: str) -> Tuple[str]:
        '''
            examples: 
                chunk='photograph'
                words= ['p', 'h', 'o', 't', 'o', 'g', 'r', 'a', 'p', 'h</w>']
        '''
        words = list(chunk)
        words[-1] += "</w>"

        while len(words) > 1:
            # valid_pairs = [pair for pair in pairwise(words) if pair in self.merges]
            valid_pairs=[]
            for pair in pairwise(words):
                if pair in self.merges:
                    valid_pairs.append(pair)
                    
            if not valid_pairs:
                break

            #测试代码
            '''
            chunk='photograph'
            ls=[(('p', 'h'), 233), (('h', 'o'), 94),
            (('o', 't'), 1351), (('t', 'o'), 68), 
            (('o', 'g'), 11030), (('g', 'r'), 197),
            (('r', 'a'), 47), (('a', 'p'), 176), 
            (('p', 'h</w>'), 1530)]
            '''
            ls=[]
            for pair in valid_pairs:
                ls.append((pair,self.merges[pair]))

            # 序号越小，出现的频率越高
            bigram = min(valid_pairs, key=lambda pair: self.merges[pair]) 
            first, second = bigram
            
            new_words = []
            for word in words:
                if word == second and new_words and new_words[-1] == first:
                    new_words[-1] = first + second
                else:
                    new_words.append(word)
            words = new_words
        print(words)
        return tuple(words)