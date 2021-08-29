# UAM-vertiport-location-selection
## 라이브러리설치
```
pip install sklearn
```
```
pip install numpy
```
```
pip install pandas
```
```
pip install matplotlib
```
## 라이브러리 가져오기
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples,silhouette_score
from sklearn.mixture import GaussianMixture
import requests 
from urllib.parse import urlparse
import numpy as np
```
## 데이터경로설정 
### 압축파일에 있는 최종데이터 경로를 입력
### 경로 입력 안하면 에러
```
data_set=pd.read_excel('최종데이터 경로 입력')
```
### 가중치 총합 값 상위 30%를 선정하기 위해 가중치 총합 값을 기준으로 내림차순 정렬
```
data_set=data_set.sort_values(by='가중치',ascending=False)
data_high20=data_set.copy()
```
### 비행 불가능 지역을 제거 안한 결과도 보여주기 위해 비행 불가능 지역 제거 안한 상태에서 상위 30% 뽑음
```
data_high20=data_set.copy()
high_20=int(len(data_high20.iloc[:,0])*0.3) ## 상위 30% 뽑음 
print(high_20)
data_high20=data_high20.iloc[0:high_20,]
data_result_able=data_high20.copy() ## 비행 불가능 지역 제거 안함
```
### 비행 불가능 지역 제거 후 상위 30% 뽑음
```
data_result_en=data_set.copy() ## 비행 불가능 지역 제거
data_result_en=data_result_en[data_result_en['비행불가능여부']==0]

data_high20_en=data_result_en.copy()
high_20=int(len(data_high20_en.iloc[:,0])*0.3) ## 상위 30% 뽑음 

print(high_20)

data_high20_en=data_high20_en.iloc[0:high_20,]
data_result_en=data_high20_en.copy()
```
## Silhouette 평가 지표 함수화
#### Silhoueette 계수를 평가하기 위해 데이터를 인자로 받음
#### Cluster label 부분과 데이터 부분을 나누어 실루엣 계수 측정
```
def eval_s(data_set):
    data=data_set.loc[:,['LAT','LON']]
    average_score=silhouette_score(data,data_set['cluster']) ## 입력된 데이터 셋을 cluster label 부분과 데이터 부분으로 나누어 성능평가
    return average_score
```

## 비행 불가능 지역 제외 안하고 K-means
#### 백업 파일 만들떄 사용이라고 주석 달아놓은 부분
#### 왼쪽에 주석 없애고 파일 경로 설정해야 에러 안남
```
data_result_able_kmeans=data_result_able.copy()
data_result_able_kmeans=data_result_able_kmeans.loc[:,['LAT','LON']]

kmeans=KMeans(n_clusters=52,init='k-means++',max_iter=300,random_state=0)
kmeans.fit(data_result_able_kmeans)

data_result_able_kmeans['cluster']=kmeans.labels_
score=eval_s(data_result_able_kmeans)


print('score=',score)
data_result_able_kmeans_plt=data_result_able_kmeans.iloc[:,0:2]
score_samples=silhouette_samples(data_result_able_kmeans_plt,data_result_able_kmeans['cluster'])

data_result_able_kmeans['silhouette-coeff']=score_samples
## 클러스터별 실루엣 계수 표준편차, 분산 확인하기 위해 태블로로 시각화 -> 파일로 저장후 태블로에서 염
g=data_result_able_kmeans[['cluster','silhouette-coeff']].groupby(['cluster']).mean().sort_values(by='silhouette-coeff')


#################################백업파일 만들때 사용########################################
#g.to_excel('분산표준편차 태블로 시각화용 경로설정')
#data_result_able_kmeans.to_excel('군집화결과경로설정')
```
