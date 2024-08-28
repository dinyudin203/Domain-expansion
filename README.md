# 영역 전개

📢 2024년 여름학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다
🎉 2024년 여름학기 AIKU Conference 열심히상 수상!

## 소개

> 고죠 사토루와 두근두근 연애 시뮬레이션💗
>

이번 프로젝트는 인공지능 언어 모델(LLM)을 활용하여 인기 만화 '주술회전'의 캐릭터 고죠 사토루처럼 말하는 AI를 개발하는 것을 목표로 합니다. 팀원들은 각기 다른 언어 모델을 선택하여, 고죠 사토루의 대화 스타일을 학습시키고, 이를 바탕으로 자연스러운 대화를 생성할 수 있는 모델을 구축했습니다. 이를 통해, 특정 캐릭터의 말투와 성격을 재현하는 AI의 가능성을 탐구하며, 최종적으로 이러한 모델을 실제 웹 서비스 형태로 배포하여 사용자들이 직접 경험할 수 있도록 했습니다.

## 방법론

- 고죠 사토루처럼 자연스럽게 대화하도록 LLM 파인튜닝
- 파인튜닝된 모델을 API 형태로 배포
- 사용자가 직접 AI와 대화할 수 있는 웹 서비스를 구축하여 배포

### 모델링

- LLaMa3-8B
- SOLAR-10.7B
- Gemma-2B

### 훈련

RAM 및 GPU 성능 이슈로 QLoRA 사용

- LLaMa3-8B + GTX Titan 8개
    - 10 epoch
- SOLAR-10.7B + Colab A100
    - 10 epoch
- Gemma-2B + Colab A100
    - 100 epoch

### 결과
- 라마 사토루
    - [https://llama3-8b-ko-gojo-demo.vercel.app](https://llama3-8b-ko-gojo-demo.vercel.app/)


- 솔라 사로투
    - https://solar-gojo.vercel.app/


- 젬마 사토루

## 사용 방법
모델 별 사용법은 각 폴더를 참고해주세요.

## 팀원
- [정우성*](https://github.com/mung3477): 아이디어 제안, LLaMa3-8b fine-tuning, Demo 개발
- [김은진](https://github.com/eunbob): Solar-10.7B fine-tuning, Demo 개발
- [성유진](성유진의 github link): Gemma-2B fine-tuning, Demo 개발
