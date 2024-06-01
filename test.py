import json
import random
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

# Load tokenizer
text_decoder_id = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(text_decoder_id)

class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths)
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)

def get_dataset(json_data_path: str = "NLP/train_dataset.json", split: str = "train"):
    with open(json_data_path) as f:
        dataset = [json.loads(line.strip()) for line in f]
        if split == "train":
            dataset = dataset[:int(len(dataset) * 0.9)]
        else:
            dataset = dataset[int(len(dataset) * 0.9):]

    template = "title : {title}\n{doc}\n\n"
    dataset_dict = [{"prompt": template.format(title=dt['title'], doc=dt['판시사항']), "label": dt['결론']} for dt in dataset]
    dataset = Dataset.from_list(dataset_dict)

    def tokenize_and_label(sample):
        encoded_prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)[:256]
        encoded_label = tokenizer.encode(sample["label"] + tokenizer.eos_token, add_special_tokens=False)[:256]

        input_ids = encoded_prompt + encoded_label
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(encoded_prompt) + encoded_label

        # 디버깅 출력
        if len(input_ids) != len(attention_mask) or len(input_ids) != len(labels):
            print(f"Length mismatch found: input_ids={len(input_ids)}, attention_mask={len(attention_mask)}, labels={len(labels)}")
            print(f"encoded_prompt: {encoded_prompt}")
            print(f"encoded_label: {encoded_label}")

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return sample

    text_decoder_input = dataset.map(tokenize_and_label, remove_columns=list(dataset.features))
    return text_decoder_input

def get_dataloader(batch_size: int = 1, split: str = "train"):
    dataset = get_dataset(split=split)

    def get_dataloader_kwargs(batch_size, dataset, tokenizer, mode):
        kwargs = {}
        kwargs["shuffle"] = False
        kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
        return kwargs

    kwargs = get_dataloader_kwargs(batch_size, dataset, tokenizer, split)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        pin_memory=True, 
        **kwargs
    )

# 디버깅을 위해 데이터셋을 불러오고 첫 몇 개의 샘플을 출력합니다.
if __name__ == "__main__":
    dataset = get_dataset("NLP/train_dataset.json", "train")
    print("First few samples in the dataset:")
    for i, sample in enumerate(dataset):
        if i < 5:
            print(sample)

    dataloader = get_dataloader(batch_size=2, split="train")
    for batch in dataloader:
        print(batch)
        break
