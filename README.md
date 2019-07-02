# chatbot  
### 1. 폴더구조  

    chatbot
    ├── generator
    │   ├── asset
    │   │   ├── attention_img
    │   │   └── training_checkpoint
    │   │       └── service
    │   ├── data
    │   │   └── KorQuAD_v1.0_train.csv
    │   ├── generator.py
    │   └── generator_train.py
    ├── line_bot
    │   ├── intent
    │   │   └── training_checkpoint
    │   │       └── service
    │   │       └── train_data.csv
    │   ├── intet_train.py
    │   ├── ner
    │   │   └── training_checkpoint
    │   │       └── service
    │   │       └── train_data.csv
    │   ├── ner_train.py
    │   └── line_bot.py
    ├── requirements.txt
    └── ner_category : ner prediction 예측값 설명
        
### 2. 개발 환경  
- python 3.6.6 에서 개발  
- 추가 라이브러리 설치 : pip install -r ./requirements.txt  

### 3. 실행  
- 각 모듈(intetn, ner, generator) 별 기 학습 된 가중치 다운(google driver : https://drive.google.com/open?id=1rR5MKNYA8qMKIBXTjuaM-k--ho-B1CoN) 후 각 모듈별 서비스 경로(/training_checkpoint/service)로 복사/ 토이 프로젝트를 목적으로 학습한 가중치이기 때문에 본인이 준비한 데이터를 통해 학습 한 가중치 사용 필요  
- ngrok 을 사용하여(https://dol2156.tistory.com/515) /chatbot/line_bot/line-hook.py 를 통해 실행되는 Flask 웹 서비스(localhost:5000)를 line channel 과 통신 가능 하도록 호스팅  
- /chatbot/line_bot/line-hook.py(line channel 과 실제로 통신을 하는 웹 서비스, intent 분류, ner 분석 모듈도 함께 기동 됨) 실행 / 실행 명령 : python line-hook.py  
- /chatbot/generator/generator.py 실행(line-hook 에서 intent 분류 결과가 질의 일 경우 호출하는 Flask 웹서비스) / 실행 명령 :  python generator.py)  


### 4. 학습  
- 각 모듈 폴더 내 ~train.py 파일(intent_train.py, ner_train.py, generator.py)을 실행시켜(실행 명령 : python intent_train.py) 학습 진행  
- 각 모듈 폴더 내 train_data.csv 파일의 포멧을 참고하여 학습 데이터 재구성  
