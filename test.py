# with open('test.txt', encoding='utf-8') as f:
#     lines = f.read().split('\n')
#     lines = lines[1:-1]
#     print(lines)
#     import pdb;pdb.set_trace()
#     merges = {tuple(bigram.split()): i for i, bigram in enumerate(lines)}
#     print(merges)

# if __name__ == '__main__':
#     for i in range(5):
#         print(chr(i))


# import unicodedata

# # 示例字符
# char = 'A'
# category = unicodedata.category(char)
# print(f"Character: {char}, Category: {category}")

# char = '1'
# category = unicodedata.category(char)
# print(f"Character: {char}, Category: {category}")

# char = '\n'
# category = unicodedata.category(char)
# print(f"Character: {repr(char)}, Category: {category}")



'''+256 生成与其他普通字符不会冲突的替换字符'''
# import unicodedata

# # 创建替换表
# table = {}
# special_count = 0
# for byte in range(256):
#     category = unicodedata.category(chr(byte))
#     if category[0] not in ['C', 'Z']:
#         table[byte] = chr(byte)
#     else:
#         table[byte] = chr(special_count + 256)
#         special_count += 1

# # 打印替换表中的前几个字符
# for key in range(256):
#     print(f"Original: {key} ({chr(key)}) -> Replacement: {ord(table[key])} ({table[key]})")

# ls1=[[49406, 320, 8853, 539, 550, 18376, 6765, 320, 4558, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407]]
# ls2=[[49406, 320, 8853, 539, 550, 18376, 6765, 320, 4558, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407]]
# print(ls1==ls2)

# import re
# text=" a photograph of An astronaut\nriding\ra         horse "
# text = re.sub(r'\s+', ' ', text)
# print(text)
# print(text.strip())
# print(text.lower())

# import re
# chunk_pattern = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
# text="a photograph of an astronaut riding a horse"
# ls=re.findall(chunk_pattern, text)
# print(ls)
# text ='photograph'
# byte=text.encode('utf-8')
# # import pdb;pdb.set_trace()
# ls=[]
# for b in byte:
#     ls.append(b)
    
# print("len(text)",len(text))
# print("len(ls)",len(ls))
# import torch

# t=torch.tensor([1,2,3,4])
# print(t*2)

# import torch

# def get_time_embedding_simple(timestep, dtype):
#     # 使用较少的频率范围
#     freqs = torch.pow(10000, -torch.arange(start=0, end=5, dtype=dtype) / 5)
#     x = torch.tensor([timestep], dtype=dtype)
#     x1 = x[:, None]
#     import pdb;pdb.set_trace()
#     x2=x1* freqs[None]
#     embedding = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
#     return freqs, x, embedding  # 返回 freqs 和 x 以供调试

# # 示例输入
# timestep = 2.0  # 时间步
# dtype = torch.float32

# # 调用函数并获取结果
# freqs, x, embedding = get_time_embedding_simple(timestep, dtype)
# print("频率 freqs:", freqs)
# print("x (timestep 与频率相乘):", x)
# print("Embedding:", embedding)

# import torch

# timestep = 2.0
# x = torch.tensor([timestep], dtype=torch.float32)
# x_reshaped = x[:, None]

# print("原始张量 x:", x)
# print("原始张量形状:", x.shape)
# print("增加维度后的 x_reshaped:", x_reshaped)
# print("增加维度后的形状:", x_reshaped.shape)

# import regex as re

# pattern = re.compile(
#     r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
#     re.IGNORECASE
# )

# text = "a photograph of an astronaut riding a horse"

# tokens = pattern.findall(text)
# print(tokens)

# from stable_diffusion_pytorch import util
# import torch

# content=torch.load(util.get_file_path('ckpt/clip.pt'))

# # 打印文件内容
# print(type(content))  # 查看内容类型

# import pdb;pdb.set_trace()
# if isinstance(content, dict):
#     # 查看字典的键
#     print("Keys in the checkpoint:", content.keys())

#     # 如果包含 'state_dict'，可以进一步检查模型权重
#     if "state_dict" in content:
#         print("State dict keys:", content["state_dict"].keys())
#     else:
#         print("Checkpoint does not contain 'state_dict'.")
# else:
#     print("The checkpoint contains:", content)
# n=torch.zeros(2,3)
# print(n)

# import torch
# import torch.nn as nn
# a=torch.tensor([1,2],dtype=torch.float32)
# print(a)
# print(nn.Parameter(a))
# print(nn.parameter.Parameter(a))

# if __name__ == "__main__":
#     s="ace"
#     #s.encode("utf-8") 返回的是字节序列b'ace'
#     #对字节序列进行迭代返回的是字节值，而不是字符本身 
#     for chunk in s.encode("utf-8"):
#         print(chunk)
    

