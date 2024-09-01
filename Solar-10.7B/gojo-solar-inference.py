import modal

app = modal.App(name="Solar-Gojo")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch==2.4.0",
	"transformers==4.44.0",
	"bitsandbytes==0.43.3",
	"accelerate==0.33.0",
    "peft==0.12.0",
)

@app.cls(image=image, gpu="T4")
class GojoModel():
    
    @modal.build()
    @modal.enter()
    def load_model_tokenizer(self):
        import torch
        import sys
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Pre-trained 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained("upstage/SOLAR-10.7B-Instruct-v1.0")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            "upstage/SOLAR-10.7B-Instruct-v1.0",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # PEFT 모델(어댑터) 로드 및 연결
        model = PeftModel.from_pretrained(
            base_model,
            "std50218/gojo-finetuned-solar",
            torch_dtype=torch.float16,
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @modal.web_endpoint(method="POST", docs=True)
    def generate(
        self,
        input: dict,
    ) -> str:
        import torch
        
        max_new_tokens=100
        # 프롬프트 생성
        prompt = f"### System:\n{input}\n\n### User:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generation_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        # 응답 처리
        output = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
        response = output.split("### User:")[1].split("### Assistant:")[0].strip()
        return {"response": response}
