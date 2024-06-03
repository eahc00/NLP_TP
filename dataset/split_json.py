import json
import random

# 원본 JSON 파일 경로
input_file = '/home/eahc00/NLP/train_dataset3.json'

# 출력할 JSON 파일 경로
output_file_1 = 'train_dataset_split.json'
output_file_2 = 'valid_dataset_split.json'

# 분할 비율 (80:20 비율)
split_ratio = 0.9

# JSON 파일을 줄별로 읽어오기
with open(input_file, 'r') as file:
    lines = file.readlines()

# 데이터를 섞기
random.shuffle(lines)

# 분할할 지점 계산
split_point = int(len(lines) * split_ratio)

# 데이터를 분할
data_part1 = lines[:split_point]
data_part2 = lines[split_point:]

# 분할된 데이터를 별도의 JSON 파일로 저장
with open(output_file_1, 'w') as file:
    file.writelines(data_part1)

with open(output_file_2, 'w') as file:
    file.writelines(data_part2)

print(f'Data has been split and saved into {output_file_1} and {output_file_2}')
