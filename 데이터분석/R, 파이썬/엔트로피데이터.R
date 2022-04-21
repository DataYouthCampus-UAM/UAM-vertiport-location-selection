
# working directroy는 data폴더로 지정 혹은 path로 직접 설정해주세요.
# path는 직접 입력해주세요.

library(readxl)
library(dplyr)
library(writexl)
path = "C:/Users/madda/Desktop/데이터분석/data"

# 엔트로피 가중치 적용할 데이터셋 만들기
entropy<- read_excel(sprintf("%s/전처리데이터.xlsx",path))%>% as.data.frame()
entropy[entropy$버스 == 0,] 
entropy <- entropy[-2723,]
entropy <- entropy[,-10]
write_xlsx(entropy,sprintf("%s/entropy.xlsx",path))



