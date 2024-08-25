import modal

app = modal.App(name="LLaMa3-Gojo")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
	"torch==2.4.0",
	"transformers==4.44.0",
	"bitsandbytes==0.43.3",
	"accelerate==0.33.0"
)

@app.cls(image=image, gpu="T4")
class Model:
	@modal.build()
	@modal.enter()
	def load_model(self):
		from transformers import pipeline

		self.base_model = "mung3477/LLaMa3-Open-Ko-8B-Instruct-gojo-KOALPACA"
		self.instruct = """<|begin_of_text|>\n<|start_header_id|>system<|end_header_id|>{utterance}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n반말로 짧게 대답해줘.\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"""

		torch_dtype = self.config_dtype_and_attn()
		model, tokenizer = self.config_model_and_tokenizer(torch_dtype, self.base_model)
		pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=64)
		self.pipe = pipe

	@modal.web_endpoint(method="POST", docs=True)
	def inference(self, request: dict) -> str:
		print(request)
		result = self.pipe(self.instruct.format(utterance=request["question"]))
		a_response = result[0]['generated_text'].split("<|end_header_id|>")[-1]

		def get_utterance(a_response: str):
			splitted = list(filter(lambda s: len(s) > 1, a_response.split("\n")))
			last_words_omitted = str.join("\n", splitted[:-1])
			original = str.join("\n", splitted)
			return last_words_omitted if len(splitted) > 1 else original

		return get_utterance(a_response)

	def config_dtype_and_attn(self):
		import torch

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

	def config_model_and_tokenizer(self, torch_dtype, base_model: str):
		from transformers import (AutoModelForCausalLM, AutoTokenizer,
											BitsAndBytesConfig)
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


