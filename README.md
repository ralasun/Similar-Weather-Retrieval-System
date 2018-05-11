## Similar-Weather-Retrieval-System
>#### Similar Weather Retrieval System using Inception-v1

### 모델 개요

#### 데이터 전처리
##### 계절별 요인 및 Boundary problem을 고려하기 위해 단일 월이 아닌 전후 3개월 데이터에 대하여 온도장 및 고도장 별 최대/최소값을 이용하여 Min-Max Scaling으로 전처리를 진행함
![preprocessing](https://user-images.githubusercontent.com/37501153/39922800-d3da703c-555b-11e8-8459-3c7d149aedaf.png)

##### Boundary problem과 계절별 요인을 고려한 Min-Max Scaling으로 총 12개의 모델이 구축하여 유사일기도 검색 시 12개월을 모두 고려한 단일 모델보다 유사도가 높은 일기도를 검색 가능함
![preprocessing2](https://user-images.githubusercontent.com/37501153/39922882-4ab3a570-555c-11e8-93ca-e5c090b78e23.png)

#### GoogleNet을 이용한 Feature Extraction
##### 고도장, 온도장, 관측시간을 포함한 4차원 데이터에 대하여 동일 날짜의 일기도 80장에 대하여 pretrained parameter를 이용해 각각 feature를 추출한 뒤, 3-D tensor를 생성함
![inceptionv1](https://user-images.githubusercontent.com/37501153/39922493-4f94df20-555a-11e8-9e87-14ea928b1d52.png)

#### GoogleNet을 이용한 유사 일기도 검색
##### Boundary Problem과 계절별 요인을 고려하여 구축한 각각의 모델에 대해 37년동안 약 3450개 날짜에 대해 3-D tensor를 형성한 뒤, 입력날짜와 나머지 모든 날짜에 대해 Mean Squared Error를 계산하여 유사도가 높은 순으로 상위 20개 랭킹을 매김
![similarity](https://user-images.githubusercontent.com/37501153/39923024-ee771bba-555c-11e8-891e-c44c3c1a3646.PNG)

#### 결과1
###### AlexNet과 AutoEncoder와 Inception-v1와의 결과 

