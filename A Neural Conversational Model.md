---
tags:
- NLP
- ML/DL
- Chatbots
---

# A Neural Conversational Model

- 논문 링크: https://arxiv.org/abs/1506.05869



## Introduction

- RNN을 이용해 이전 시퀀스에 대응하는 시퀀스를 예측하는 모델을 구축.
-  IT helpdesk 데이터셋과 노이즈가 섞인 영화 대사에 대해 실험한 결과 n-gram보다 더 좋은 성능을 보임



## Model

- Sutskever et al., 2014가 제안한 seq2seq 모델을 사용함.
- 학습 중 진짜 출력 시퀀스가 주어지지 않은 상황에서 예측된 출력을 다음 출력의 입력값으로 넣음 (greedy inference). 덜 탐욕적인 접근은 beam 검색을 사용했을 것이며 여러 후보를 다음 출력의 입력으로 넣었을 것이다.
- 모델의 강점은 간단함에 있음. MT, QA, 대화 등에 큰 변화 없이 사용될 수 있음
- 그러나 번역과 달리 seq2seq 모델은 대화를 모델링 하는데 있어 문제를 "해결"하는 데는 성공적이지 않을 것임. 이는 몇 가지 단순화 때문인데, 목적 함수가 실제 인간의 대화를 포착하지 못함. 인간의 대화는 단순히 다음 단계의 예측이 아니라 긴 대화와 정보 교환에 의해 이뤄지기 때문
- 또한 일관성과 일반적인 지식의 부족함도 비지도 모델의 한계점



## Datasets

- IT Helpdesk Troubleshooting dataset
- OpenSubtitles dataset



## Discussion

- 해당 모델은 간단하고 기본적인 대화를 생성해내며 노이즈가 섞인 오픈 도메인 데이터에서 지식을 뽑아낼 수 있음
- 간단한 모델이지만 어떠한 규칙도 없이 데이터에만 의존한 모델이 여러 타입의 질문에 적절한 답변을 생성해낸 것은 놀라운 일
- 그러나 일관되지 않은 성격 탓에 Turing test는 통과하지 못함.