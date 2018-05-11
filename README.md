## Similar-Weather-Retrieval-System
>#### Similar Weather Retrieval System using Inception-v1

### 모델 개요

#### 데이터 전처리
##### 계절별 요인 및 Boundary problem을 고려하기 위해 단일 월이 아닌 전후 3개월 데이터에 대하여 온도장 및 고도장 별 최대/최소값을 이용하여 Min-Max Scaling으로 전처리를 진행함
![preprocessing](https://user-images.githubusercontent.com/37501153/39922800-d3da703c-555b-11e8-8459-3c7d149aedaf.png)

##### Boundary problem과 계절별 요인을 고려한 Min-Max Scaling으로 총 12개의 모델이 구축하여 유사일기도 검색 시 12개월을 모두 고려한 단일 모델보다 유사도가 높은 일기도를 검색 가능함
![inceptionv1](https://user-images.githubusercontent.com/37501153/39922493-4f94df20-555a-11e8-9e87-14ea928b1d52.png)

>> I

