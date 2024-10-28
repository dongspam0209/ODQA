# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Project - Open-Domain Question Answering

## Installation and Quick Start
**Step 1.** 프로젝트에 필요한 모든 dependencies는 `requirements.txt`에 있고, 이에 대한 가상환경을 생성해서 프로젝트를 실행
```sh
# 가상환경 만들기
$ python -m venv .venv

# 가상환경 켜기
$ . .venv/bin/activate

# 제공되는 서버 환경에 따라 선택적 사용
$ export TMPDIR=/data/ephemeral/tmp 
$ mkdir -p $TMPDIR

# 필요 라이브러리 설치
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

**Step 2.** Pre Processing 실행
 ```sh
# 작업환경 변경
$ cd pre_process

# 다음 주피터를 따라가며 KorQuAD 1.0 데이터 증강
$ data_augment_korquadv1.ipynb

# 다음 주피터를 따라가며 AIHub 데이터 증강
$ data_augment_aihub.ipynb

# 다음 주피터를 따라가며 DPR retrieval을 학습하기 위한 데이터 만들기
$ generate_DPR_dataset_korquad.ipynb
```

**Step 3.** DPR 모델 학습

**utils/arguments_dpr.py** 에서 DPR 학습을 위한 파라미터 변경
- model : 원하는 사전 학습된 모델 불러오기
- train_data : generate_DPR_dataset_korquad.ipynb 에서 생성한 데이터 경로
- valid_data : generate_DPR_dataset_korquad.ipynb 에서 생성한 데이터 경로
- q_output_path : Query embedding 모델 저장할 경로
- c_output_path : Context embedding 모델 저장할 경로
```sh
# ./level2-mrc-nlp-08 경로에서 실행
$ python train_dpr.py
```



**Step 4.** Retrieval를 위한 사전처리 진행

**database/python get_embedding_vec.csv** : BM25 모델 및 DPR의 embedding vector 저장
- model : 학습된 context embedding 모델 경로
- wiki_path : Wiki.doc 데이터 경로 
- valid_data : Query-Passage 쌍 데이터의 검증 데이터 경로
- save_path : Embedding vector 저장 경로

**test_retrieval.py**
- model : 학습된 query embedding 모델 경로
- valid_data : Query-Passage 쌍 데이터의 검증 데이터 경로
- faiss_path : **database/python get_embedding_vec.csv** 에서 실행한 save_path 경로
- bm25_path : **database/python get_embedding_vec.csv** 에서 실행한 save_path 경로
- context_path : **database/python get_embedding_vec.csv** 에서 실행한 save_path 경로

**test_retrieval_inference.py**
- model : 학습된 query embedding 모델 경로
- test_dataset : Query-Passage 쌍 데이터의 테스트 데이터 경로
- faiss_path : 위와 동일
- bm25_path : 위와 동일
- context_path : 위와 동일

```sh
# 작업환경 변경
$ cd database

# 다음 코드를 실행하여 embedding vector 추출
$ python get_embedding_vec.csv

# BM25 및 DPR 성능 확인
$ cd ..
$ python test_retrieval.py

# Inference 시 사용할 retireve 된 데이터 생성
$ python test_retrieval_inference.py
```


**Step 5.** Reader 학습

**utils/arguments_extraction_reader.py**에서 extracion based model 학습을 위한 파라미터 변경
- model_name_or_path : 사전 학습된 모델 불러오기
- dataset_name : Query-Passage 쌍 데이터나 증강된 데이터 경로로 변경
- output_dir : 학습된 모델 및 평가 결과 저장 경로

```sh
# 다음 코드를 실행하여 extraction based model 학습
$ python train_extraction_reader.py

# 프로젝트 때는 사용하지 않았지만 generation based model 학습, 파라미터 변경은 위와 동일
$ python train_generation_reader_Seq2SeqLM,.py
$ python train_generation_reader_CausalLM,.py
```



**Step 6.** Inference 실행

**utils/arguments_inference.py**에서 inference 할 extraction based 모델의 파라미터 변경
- model_name_or_path : 학습이 완료된 모델 불러오기
- output_dir : Inference 결과 저장 경로

```sh
# 코드 50번째 줄에서 retireve 된 데이터 불러오는 경로 원하는 것으로 변경하면서 사용
$ python inference.py
```


**Step 7.** 앙상블 실행
```sh
# train_extraction_reader 실행 시 생성되는 predictions.json 값들로 상관분석 분석
$ correlation_exp.ipynb

# 상관분석을 통해 사용할 모델 선택 되었다면 그 모델들로 inference 시 생성된 nbest_predictions.json 파일들로 앙상블 진행 / 두가지 버전 모두 사용 가능
$ ensemble_v1.ipynb
$ ensemble_v2.ipynb

```