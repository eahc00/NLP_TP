from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, default_data_collator
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.data import DataLoader, BatchSampler
from transformers.data import DataCollatorForSeq2Seq
import random
import numpy as np

# import nltk
# from nltk.tokenize import sent_tokenize
# nltk.download('punkt')

text_decoder_id = f"MLP-KTLim/llama-3-Korean-Bllossom-8B"
tokenizer = AutoTokenizer.from_pretrained(
    text_decoder_id,
)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if not data_source:
            raise ValueError("data_source is empty.")
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



def get_dataset(
        json_data_path:str = "./train_dataset3.json",
        split:str = "train",
        get_prompy_only = False
        ):
    """
    split parameter can be: train, valid, test
    TODO return appropriate dataset according to split
    TODO get documents,summaries according to split
    """
    

    template = """
        system : 한국말로만 대답하고 최대한 간결하고 알기쉽게 정리해줘.
        user : 사건의 title과 판시사항을 보고 판결 결과와 그 이유를 예측해줘
        \n title:{title}\n판시사항:{doc}
        assistant : 
    """

    # dataset_path = f"./../dataset/{split}.dataset" ## You should make this dataset as jsonl file
    if split == "test" :
        json_data_path = "./test_dataset.json"
    with open(json_data_path) as f:
        import json
        dataset = [json.loads(line.strip()) for line in f]
        if split == "train" :
            dataset = dataset[:int(len(dataset) * 0.9)]
        elif split == "valid": 
            dataset = dataset[int(len(dataset) * 0.9):]
        


    # dataset_dict = [{"prompt":template.format(doc=dt['doc']),"label":dt['summary']} for dt in dataset]
    if split == "test" :
        dataset_dict = [{"prompt":template.format(title=dt['title'], doc=dt['판시사항'])} for dt in dataset]
        return dataset_dict

    dataset_dict = [{"prompt":template.format(title=dt['title'], doc=dt['판시사항']), "label":dt['결론']} for dt in dataset]

    if get_prompy_only :
        return dataset_dict
    print("dataset_dict: ", dataset_dict[0])
    dataset = Dataset.from_list(dataset_dict)


    def tokenize_and_label(sample):
        if "prompt" not in sample or "label" not in sample:
            raise ValueError("Sample missing required keys: 'prompt' or 'label'.")
        encoded_prompt= tokenizer.encode(tokenizer.bos_token + sample["prompt"],add_special_tokens=False)[:512]
        encoded_label = tokenizer.encode(sample["label"]+tokenizer.eos_token, add_special_tokens=False)[:512]

        sample = {
            "input_ids":encoded_prompt + encoded_label,
            "attention_mask":[1]*(len(encoded_prompt)+len(encoded_label)),
            "labels":[-100]*len(encoded_prompt) + encoded_label,
        }
        return sample

    text_decoder_input = dataset.map(tokenize_and_label, remove_columns=list(dataset.features))
    return text_decoder_input


def get_dataloader(
    batch_size:int = 1, 
    split:str="train"
    ):
    """
    Warning: batch_size should be same as stable diffusion dataloader batch size
    """


    dataset = get_dataset(split=split)

    def get_dataloader_kwargs(batch_size, dataset, tokenizer, mode):
        kwargs = {}
        kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode==split)
        kwargs["shuffle"] = False
        # kwargs["collate_fn"] = default_data_collator()
        kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
        return kwargs
    
    kwargs = get_dataloader_kwargs(batch_size, dataset, tokenizer, split)

    return torch.utils.data.DataLoader(
        dataset,
        # batch_size = 1, 
        pin_memory=True, 
        **kwargs
        )
    


