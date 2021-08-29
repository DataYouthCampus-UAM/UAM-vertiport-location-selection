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
## 비행 불가능 지역 제외 하고 K-means
#### 백업 파일 만들떄 사용이라고 주석 달아놓은 부분
#### 왼쪽에 주석 없애고 파일 경로 설정해야 에러 안남
```
data_result_en_kmeans=data_result_en.copy()
data_result_en_kmeans=data_result_en_kmeans.loc[:,['LAT','LON']]
kmeans=KMeans(n_clusters=52,init='k-means++',max_iter=300,random_state=0)
kmeans.fit(data_result_en_kmeans)


data_result_en_kmeans['cluster']=kmeans.labels_

score=eval_s(data_result_en_kmeans)

print('score=',score)

data_result_en_kmeans_plt=data_result_en_kmeans.iloc[:,0:2]

score_samples=silhouette_samples(data_result_en_kmeans_plt,data_result_en_kmeans['cluster'])
data_result_en_kmeans['silhouette-coeff']=score_samples
## 클러스터별 실루엣 계수 표준편차, 분산 확인하기 위해 태블로로 시각화 -> 파일로 저장후 태블로에서 염
g=data_result_en_kmeans[['cluster','silhouette-coeff']].groupby(['cluster']).mean().sort_values(by='silhouette-coeff')


##################################백업팡리 만들때 사용####################################
#g.to_excel('태블로용 분산, 표준편차 데이터 백업 경로')
#data_result_en_kmeans.to_excel('클러스터링 결과 백업 경로')
```
## 분산,표준편차 확인
```
print(np.var(g),np.std(g)) ## 표준편차 , 분산 확인
```

##  DBSCAN용 함수
```
list_array=np.arange(1,0,-0.001)
## eps 파라미터 용 배열 생성
```

#### 파라미터 튜닝 용 함수
```
 #파라미터 튜닝 용 함수 , 데이터와 eps,min_samples 인자를 받고 클러스터 개수를 return함
def db_scan(dbscan_data,ep,samples):
    dbscan=DBSCAN(eps=ep,min_samples=samples,metric='euclidean')
    dbscan_labels=dbscan.fit_predict(dbscan_data)
    len_cluster=np.unique(dbscan_labels)
    num_cluster=np.max(len_cluster)
    return num_cluster
```



## 비행 불가능 지역 제외 안하고 DBSCAN
#### 백업 파일 만들떄 사용이라고 주석 달아놓은 부분
#### 왼쪽에 주석 없애고 파일 경로 설정해야 에러 안남
```
parameter_tuning_min_samples=[]
parameter_tuning_eps=[]
data_result_able_dbscan=data_result_able.copy()
data_result_able_dbscan=data_result_able_dbscan.loc[:,['LAT','LON']]
score_list=[]
########################parameter tuning##################
for samples in range(1,8,1):
    for ep in list_array:
        len_cluster=db_scan(data_result_able_dbscan,ep,samples)
        if len_cluster!=51:
            pass
        else :
            parameter_tuning_min_samples.append(samples)
            parameter_tuning_eps.append(ep)
            
################################################################
#######################튜닝된 파라미터별 Score 확인###################################
for w, q in zip(parameter_tuning_min_samples,parameter_tuning_eps):
    dbscan=DBSCAN(eps=q,min_samples=w,metric='euclidean')
    dbscan_labels=dbscan.fit_predict(data_result_able_dbscan)
    dbscan_data=data_result_able_dbscan.copy()
    dbscan_data['cluster']=dbscan_labels
    dbscan_data=dbscan_data[dbscan_data['cluster']!=-1] #outlier 제거 
    score=eval_s(dbscan_data)
    score_list.append(score)    
    
max_score=score_list.index(np.max(score))#가장 높은 score 파라미터 선택
print(max_score)
a=parameter_tuning_eps[max_score]
b=parameter_tuning_min_samples[max_score]

###########################################################################################

dbscan=DBSCAN(eps=a,min_samples=b,metric='euclidean')
dbscan_labels=dbscan.fit_predict(data_result_able_dbscan)
dbscan_data=data_result_able_dbscan.copy()
dbscan_data['cluster']=dbscan_labels
dbscan_data=dbscan_data[dbscan_data['cluster']!=-1] # outlier 제거

############클러스터링 결과 데이터 백업 파일 경로 설정###############################
#dbscan_data.to_excel('C:\\Users\\qjawl\\Desktop\\빅데이터\\빅데이터프로젝트\\UAM\\20210827\\dbscan\\dbscan_able.xlsx')
#################################################################
##########################evaluation##################################

#dbscan_data.drop(['dbscan_labels'],axis=1,inplace=True)
score=eval_s(dbscan_data)
print('score=',score)
dbscan_data_plt=dbscan_data.iloc[:,0:2]
score_samples=silhouette_samples(dbscan_data_plt,dbscan_data['cluster'])
dbscan_data['silhouette-coeff']=score_samples

## 클러스터별 실루엣 계수 표준편차, 분산 확인하기 위해 태블로로 시각화 -> 파일로 저장후 태블로에서 염
g=dbscan_data[['cluster','silhouette-coeff']].groupby(['cluster']).mean().sort_values(by='silhouette-coeff')



#######태블로용 데이터 백업 경로 설정 #########
#g.to_excel('C:\\Users\\qjawl\\Desktop\\빅데이터\\빅데이터프로젝트\\UAM\\20210829\\평균분산표준편차\\kmeans.xlsx')
```


## 비행 불가능 지역 제외 하고 DBSCAN
#### 백업 파일 만들떄 사용이라고 주석 달아놓은 부분
#### 왼쪽에 주석 없애고 파일 경로 설정해야 에러 안남

```
parameter_tuning_min_samples=[]
parameter_tuning_eps=[]
data_result_en_dbscan=data_result_en.copy()
data_result_en_dbscan=data_result_en_dbscan.loc[:,['LAT','LON']]
score_list=[]
########################parameter tuning##################
for j in range(1,5,1):
    print(j,'^'*100)
    for i in list_array:
        len_cluster=db_scan(data_result_en_dbscan,i,j)
        print(len_cluster)
        if len_cluster==51:
            print('51개 minsamples',j)
            parameter_tuning_min_samples.append(j)
            parameter_tuning_eps.append(i)
        else:
            pass
################################################################
#######################튜닝된 파라미터 별 score 확인##################################

for a, b in zip(parameter_tuning_min_samples,parameter_tuning_eps):
    dbscan=DBSCAN(eps=b,min_samples=a,metric='euclidean')
    dbscan_labels=dbscan.fit_predict(data_result_en_dbscan)
    dbscan_data=data_result_en_dbscan.copy()
    dbscan_data['cluster']=dbscan_labels
    dbscan_data=dbscan_data[dbscan_data['cluster']!=-1]
    score=eval_s(dbscan_data)
    score_list.append(score)
    
max_score=score_list.index(np.max(score_list))#가장 높은 score 파라미터 선택
a=parameter_tuning_eps[max_score]
b=parameter_tuning_min_samples[max_score]
###################################################################################



dbscan=DBSCAN(eps=a,min_samples=b,metric='euclidean')
dbscan_labels=dbscan.fit_predict(data_result_en_dbscan)
dbscan_data=data_result_en_dbscan.copy()
dbscan_data['cluster']=dbscan_labels
dbscan_data=dbscan_data[dbscan_data['cluster']!=-1]

############클러스터링 결과 데이터 백업 파일 경로 설정###############################
#dbscan_data.to_excel('C:\\Users\\qjawl\\Desktop\\빅데이터\\빅데이터프로젝트\\UAM\\20210829\\dbscan\\dbscan_en.xlsx')

#################################################################
##########################evaluation##################################
try:
    dbscan_data.drop(['dbscan_labels'],axis=1,inplace=True)
except:
    pass

score=eval_s(dbscan_data)
print('score=',score)

dbscan_data_plt=dbscan_data.iloc[:,0:2]

score_samples=silhouette_samples(dbscan_data_plt,dbscan_data['cluster'])
dbscan_data['silhouette-coeff']=score_samples

## 클러스터별 실루엣 계수 표준편차, 분산 확인하기 위해 태블로로 시각화 -> 파일로 저장후 태블로에서 염
g=dbscan_data[['cluster','silhouette-coeff']].groupby(['cluster']).mean().sort_values(by='silhouette-coeff')


#######태블로용 데이터 백업 경로 설정 #########
##g.to_excel('C:\\Users\\qjawl\\Desktop\\빅데이터\\빅데이터프로젝트\\UAM\\20210829\\평균분산표준편차\\dbscan.xlsx')
```

## 분산 표준편차 확인
```
print(np.var(g),np.std(g))
```        


