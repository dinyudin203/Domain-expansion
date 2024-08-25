# https://velog.io/@judy_choi/LLaMA3-%EC%9E%90%EC%98%81%EC%97%85-QA-%EC%B1%97%EB%B4%87-Fine-Tuning

import argparse
import os

import huggingface_hub
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments, logging,
                          pipeline)
from trl import SFTTrainer

# Hugging Face Basic Model 한국어 모델
base_model = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"	# beomi님의 Llama3 한국어 파인튜닝 모델
qna_dataset = "Chat-Satoru-all-LLaMa.json"
new_model = "LLaMa3-Open-Ko-8B-Instruct-gojo-KOALPACA"

instruct = """<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>{utterance}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n반말로 짧게 대답해줘.\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"""

def config_dtype_and_attn():
	# 현재 사용중인 GPU의 CUDA 연산 능력을 확인한다.
	# 8이상이면 고성능 GPU 로 판단한다.
	if torch.cuda.get_device_capability()[0] >= 8:
		# 고성능 Attention인 flash attention 2 을 사용
		# attn_implementation = "flash_attention_2"
		# 데이터 타입을 bfloat16으로 설정해준다.
		# bfloat16은 메모리 사용량을 줄이면서도 계산의 정확성을 유지할 수 있는 데이터 타입이다.
		torch_dtype = torch.bfloat16
	else:
		# attn_implementation = "eager"
		torch_dtype = torch.float16

	print(f"torch data type: {torch_dtype}")
	# return attn_implementation, torch_dtype
	return torch_dtype

def config_model_and_tokenizer(torch_dtype, base_model: str):
	# QLoRA config
	quant_config = BitsAndBytesConfig(
		load_in_4bit=True,	# 모델 가중치를 4비트로 로드
		bnb_4bit_quant_type="nf4",	# 양자화 유형으로는 “nf4”를 사용한다.
		bnb_4bit_compute_dtype=torch_dtype,	# 양자화를 위한 컴퓨팅 타입은 직전에 정의 했던 torch_dtype으로 지정 해준다.
		bnb_4bit_use_double_quant=False,	# 이중 양자화는 사용하지 않는다.
	)

	# 모델 로드
	model = AutoModelForCausalLM.from_pretrained(
	    base_model,
	    quantization_config=quant_config,
	    # device_map={"": 0}	# 0번째 gpu 에 할당
	    device_map="auto"
	)
	# 모델의 캐시 기능을 비활성화 한다. 캐시는 이전 계산 결과를 저장하기 때문에 추론 속도를 높이는 역할을 한다. 그러나 메모리 사용량을 증가시킬 수 있기 때문에, 메모리부족 문제가 발생하지 않도록 하기 위해 비활성화 해주는 것이 좋다.
	model.config.use_cache = False
	# 모델의 텐서 병렬화(Tensor Parallelism) 설정을 1로 지정한다. 설정값 1은 단일 GPU에서 실행되도록 설정 해주는 의미이다.
	model.config.pretraining_tp = 1

	# 토크나이저 로드
	tokenizer = AutoTokenizer.from_pretrained(
	              base_model,
	              trust_remote_code=True)
	# 시퀀스 길이를 맞추기 위해 문장 끝에 eos_token를 사용
	tokenizer.pad_token = tokenizer.eos_token
	# 패딩 토큰을 시퀀스의 어느 쪽에 추가할지 설정
	tokenizer.padding_side = "right"

	return model, tokenizer

def finetune():
	dataset = load_dataset("json", data_files=qna_dataset, split="train")
	# 데이터 확인
	print(len(dataset))
	print(dataset[0])

	torch_dtype = config_dtype_and_attn()
	model, tokenizer = config_model_and_tokenizer(torch_dtype, base_model)

	peft_params = LoraConfig(
	    lora_alpha=16,	# LoRA의 스케일링 계수를 설정 한다. 값이 클 수록 학습 속도가 빨라질 수 있지만, 너무 크게 되면 모델이 불안정해질 수 있다.
	    lora_dropout=0.1,	#  과적합을 방지하기 위한 드롭아웃 확률을 설정한다. 여기서는 10%(0.1)의 드롭아웃 확률을 사용하여 모델의 일반화 성능을 향상시킨다.
	    r=64,	# LoRA 어댑터 행렬의 Rank를 나타낸다. 랭크가 높을수록 모델의 표현 능력은 향상되지만, 메모리 사용량과 학습 시간이 증가한다. 일반적으로 4, 8, 16, 32, 64 등의 값을 사용한다.
	    bias="none",	# LoRA 어댑터 행렬에 대한 편향을 추가할지 여부를 결정한다. “none”옵션을 사용하여 편향을 사용하지 않는다.
	    task_type="CAUSAL_LM",	# LoRA가 적용될 작업 유형을 설정한다. CAUSAL_LM은 Causal Language Modeling 작업을 의미한다. 이는 특히 GPT 같은 텍스트 생성 모델에 주로 사용된다.
	)

	training_params = TrainingArguments(
	    output_dir="/home/aikusrv01/aiku/Gojo/woosung/results",
	    num_train_epochs=10,	# 기본값은 3
	    per_device_train_batch_size=4,	# 기본값은 8
	    gradient_accumulation_steps=1,	# 기본값 1
	    optim="paged_adamw_32bit",
	    save_steps=25,
	    logging_steps=25,
	    learning_rate=2e-4,
	    weight_decay=0.001,
	    fp16=False,
	    bf16=False,
	    max_grad_norm=0.3,
	    max_steps=-1,
	    warmup_ratio=0.03,
	    group_by_length=True,
	    lr_scheduler_type="constant",
	    report_to="tensorboard"
	)

	trainer = SFTTrainer(
	    model=model,
	    train_dataset=dataset,
	    peft_config=peft_params,
	    dataset_text_field="text",
	    max_seq_length=64,	# 256, 512 등으로 수정할 수 있음.
	    tokenizer=tokenizer,
	    args=training_params,
	    packing=False,
	)

	trainer.train()
	trainer.save_model(new_model)

	return model, tokenizer

def test(model, tokenizer):
	logging.set_verbosity(logging.CRITICAL)
	while True:
		question = input("고죠에게 뭐라고 전달할까? : ")
		pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=64)
		result = pipe(instruct.format(utterance=question))
		a_response = result[0]['generated_text'].split("<|end_header_id|>")[-1]

		def print_last_words_omitted(a_response: str):
			splitted = list(filter(lambda s: len(s) > 1, a_response.split("\n")))
			last_words_omitted = str.join("\n", splitted[:-1])
			original = str.join("\n", splitted)
			print(last_words_omitted) if len(splitted) > 1 else print(original)

		print_last_words_omitted(a_response)




def parse_CLI_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", help="train or test", type=bool, default=False)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"

	args = parse_CLI_args()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print('Device:', device)
	print('Current cuda device:', torch.cuda.current_device())
	print('Count of using GPUs:', torch.cuda.device_count())

	load_dotenv()
	huggingface_hub.login(os.environ.get("HF_TOKEN"))

	print(f"train: {args.train}")
	if args.train:
		model, tokenizer = finetune()
		model.save_pretrained(new_model)
		tokenizer.save_pretrained(new_model)
	else:
		torch_dtype = config_dtype_and_attn()
		model, tokenizer = config_model_and_tokenizer(torch_dtype, base_model)
		model.from_pretrained("[Your username]/" + new_model)
		print("Pretrained model was loaded!")

		model.save_pretrained(new_model)
		tokenizer.save_pretrained(new_model)
		test(model, tokenizer)
