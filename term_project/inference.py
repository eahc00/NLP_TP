from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from safetensors.torch import load_file

class Application:

    __model = None
    __tokenizer = None


    @classmethod
    def _load(cls):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        text_decoder_id = f"upstage/SOLAR-10.7B-Instruct-v1.0"
        peft_model_id = f"upstage/SOLAR-10.7B-Instruct-v1.0_Lora"

        if device == "cuda" :
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # LoRA 적용 모델 불러오기
            base_model = AutoModelForCausalLM.from_pretrained(
                text_decoder_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config
            )

        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                text_decoder_id,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

        load_model = PeftModel.from_pretrained(base_model, peft_model_id)
        loaded_state_dict = load_file(
            "/home/eahc00/NLP/term_project/upstage/SOLAR-10.7B-Instruct-v1.0_Lora/adapter_model.safetensors"
        )
        load_model.load_state_dict(loaded_state_dict, strict=False)
        tokenizer = AutoTokenizer.from_pretrained(text_decoder_id)
        tokenizer.pad_token = tokenizer.eos_token

        load_model.to(device)

        cls.__model = load_model
        cls.__tokenizer = tokenizer

    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.__class__.__model is None:
            self.__class__._load()

    @property
    def model(self):
        return self.__class__.__model

    @property
    def tokenizer(self) :
        return self.__class__.__tokenizer


    def __call__(self, input):
        inputs = self.tokenizer(input, return_tensors="pt")

        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.generate(input_ids=inputs["input_ids"], max_new_tokens=448)
            response = str(self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

        return response