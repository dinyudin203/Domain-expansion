# Finetuning LLaMa3-8B
This repository provides codes to fine-tune LLaMa3-8B using 8,000 synthetic Korean chat data in [ALPACA](https://github.com/gururise/AlpacaDataCleaned?tab=readme-ov-file) format. We utilized the Gemini API to create a fictional answer from [Gojo Satoru](https://jujutsu-kaisen.fandom.com/wiki/Satoru_Gojo), a character in the japanise anime [Jujutsu Kaisen](https://jujutsu-kaisen.fandom.com/wiki/Satoru_Gojo).

## Installation
This repository was tested with python 3.12.0 and torch 2.4. You can install all requirements by using the following command.
```
pip install -r requirements.txt
```

## Train
Run the following command to train the model. You should have an environment variable `HF_TOKEN` in `.env` file to use HuggingFace API.
```
python3 LLaMa3-finetune.py --train=True
```

The code fine-tunes the model [beomi/Llama-3-Open-Ko-8B-Instruct-preview](https://huggingface.co/beomi/Llama-3-Open-Ko-8B-Instruct-preview), which has an ability to understand Korean. It was slightly modified from the code used on [this blog](https://velog.io/@judy_choi/LLaMA3-%EC%9E%90%EC%98%81%EC%97%85-QA-%EC%B1%97%EB%B4%87-Fine-Tuning).

It took approximately 20 hours to train the model on eight GTX Titans.

## Deploy
The model can be deployed using [Modal labs](https://modal.com/docs/guide/webhooks#web_endpoint). Use official tutorials to deploy your code. My code might be a useful reference.

[This](https://llama3-8b-ko-gojo-demo.vercel.app) is my demo.
