### NLP TermProject

#### Dataset
- 판례 데이터셋을 selenium을 이용한 크롤링으로 구축
- pandas를 활용하여 데이터셋 전처리.

#### Model
- LLM을 LoRA로 finetuning하여 사용
- huggingface에 있는 upstage/SOLAR-10.7B-Instruct-v1.0, MLP-KTLim/llama-3-Korean-Bllossom-8B모델을 이용.

#### Train
- PEFT(Quantization, Mixed precision, LoRA를 활용)
- epoch : 20
- optimizer : Adam
- learning rate : 2e-5 + StepLR

#### evaluation
- ROUGE score로 evaluation 진행

#### interface
- Discord chatbot으로 UI 구축.
