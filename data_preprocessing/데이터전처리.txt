# UAM
UAM_Porjcet
```
#install.packages(c("readxl","writexl","dplyr","Imap","stringr"))


library(readxl)
library(writexl)
library(dplyr)
library(Imap)
library(stringr)

# 교통혼잡,인구수 데이터 변수 추가
trans <- read_excel("./교통관련데이터.xlsx") %>% as.data.frame()
trans_copy <- trans[,-c(7,9,10)]
trans_copy$행정주소 <- paste(trans_copy$`도/광역시/시`,trans_copy$시군구,trans_copy$읍면동,
                         rep = " ")
trans_copy <- trans_copy[,-c(1,2,3)]
trans_copy <- trans_copy[,c(5,1:4)]
trans_copy
trans_copy[is.na(trans_copy),]
write_xlsx(trans_copy,"2018년_행정주소.xlsx")

# 행정주소, 법정주소 연결시키기
relation_data <- read_excel("./행정법정동코드 연계자료.xlsx")
copy_relation_data <- relation_data[,-c(8,10,11,12)]
colnames(copy_relation_data) <- copy_relation_data[1,]
copy_relation_data <- copy_relation_data[-1,]

## 읍면동 데이터만 남기기
si_data_list <- grep("시$",copy_relation_data$법정동, value = 1)
gu_data_list <- grep("구$",copy_relation_data$법정동, value = 1)
gun_data_list <- grep("군$",copy_relation_data$법정동, value = 1)
ri_data_list <- grep("리$",copy_relation_data$법정동, value = 1)
do_data_list <- grep("도$",copy_relation_data$법정동, value = 1)
copy_relation_data <- copy_relation_data[!(copy_relation_data$법정동 %in% gu_data_list),]
copy_relation_data <- copy_relation_data[!(copy_relation_data$법정동 %in% gun_data_list),]
copy_relation_data <- copy_relation_data[!(copy_relation_data$법정동 %in% si_data_list),]
copy_relation_data <- copy_relation_data[!(copy_relation_data$법정동 %in% ri_data_list),]
copy_relation_data <- copy_relation_data[!(copy_relation_data$법정동 %in% do_data_list),]
copy_relation_data$법정동코드 <- floor(as.numeric(copy_relation_data$법정동코드)/100)
copy_relation_data$행정주소 <- paste(copy_relation_data$시도,copy_relation_data$시군구,
                                 copy_relation_data$행정구역명, rep ="")
copy_relation_data$법정동명 <- paste(copy_relation_data$시도,copy_relation_data$시군구,
                                 copy_relation_data$법정동, rep ="")
names(copy_relation_data)[8] <- "EMD_CD"


## copy_relation_data 결측값 수정
copy_relation_data$행정구역코드[copy_relation_data$법정동 == "신수동" & copy_relation_data$시군구 == "사천시"] <- 0
copy2_relation_data <- na.omit(copy_relation_data) %>% as.data.frame()
sum(is.na(copy2_relation_data))
write_xlsx(copy2_relation_data,"copy2_relation_data.xlsx")


# trans_copy, copy2_relation_data 합치기
trans_copy$행정주소
trans_copy$행정주소 <- str_trim(trans_copy$행정주소, side = "right")
copy2_relation_data <- read_excel("./copy2_relation_data.xlsx")
join_data <- left_join(copy2_relation_data,trans_copy, by ="행정주소")
first_data <- join_data[,c(8:14)]
names(first_data)[3] <- "법정동주소"


# 행정주소 겹친거 제거하기
first_data$id <- c(1:length(first_data$EMD_CD))
eraser_rep_data <- first_data[duplicated(first_data$행정주소),]
no_rep_data <- first_data[!(first_data$id %in% eraser_rep_data$id),]

# 관측 안된 값 제거하기
na_data <- no_rep_data[is.na(no_rep_data$승용차),]
second_data <- no_rep_data[!(no_rep_data$id %in% na_data$id),]
write_xlsx(second_data,"2차가공데이터.xlsx")



# 위경도 추가하기
location <- read.csv("./center_location.csv")
location <- as.data.frame(location)
location <- location[,-1]
colnames(location) <- c("LON","LAT")
location <- location[,c(2,1)]

# location, v2_copy 데이터 합치기
join_data_2 <- data.frame(second_data,location)
sum(is.na(join_data_2))

# 읍면동 중심위치와 가까운 터미널 거리 구하기
terminal <- read_excel("./터미널_위키.xlsx")
terminal <- terminal[,-c(1,4)]
colnames(terminal) <- c("LON","LAT")
terminal_dist <- c()
for(i in c(1:length(join_data_2$LON))){
  data <- c()
  data <- gdist(lon.1 = join_data_2$LON[i],
                lat.1 = join_data_2$LAT[i],
                lon.2 = terminal$LON,
                lat.2 = terminal$LAT,
                units = "m")
  terminal_dist <- c(terminal_dist, min(data))
}


# 읍면동 중심위치와 가까운 기차역 거리 구하기
train_station_data <- read_excel("./기차역좌표.xlsx")
train_station_data <- train_station_data[,-1]
train_station_dist <- c()

for(i in c(1:length(join_data_2$LON))){
  train_data <- c()
  train_data <- gdist(lon.1 = join_data_2$LON[i],
                      lat.1 = join_data_2$LAT[i],
                      lon.2 = train_station_data$경도,
                      lat.2 = train_station_data$위도,
                      units = "m")
  train_station_dist <- c(train_station_dist,min(train_data))
}


# 읍면동 중심위치와 가까운 공항 거리 구하기
airport_data <- read.csv("./공항위치.csv")
colnames(airport_data) <- c("공항","위도","경도")
airport_data <- airport_data[,-1]
airport_dist <- c()

for(i in c(1:length(join_data_2$LON))){
  airport <- c()
  airport <- data <- gdist(lon.1 = join_data_2$LON[i],
                           lat.1 = join_data_2$LAT[i],
                           lon.2 = airport_data$경도,
                           lat.2 = airport_data$위도,
                           units = "m")
  airport_dist <- c(airport_dist,min(airport))
}

# 3개 거리 합치기
distance <- data.frame(terminal_dist,train_station_dist,airport_dist)
colnames(distance) <- c("터미널과의거리","기차역과의거리","공항과의거리")


# 3차 데이터 만들기 : join_data_2와 distance 데이터셋 합치기
third_data <- data.frame(c(0:3269),join_data_2,distance)
third_data <- third_data[,c(1,10:11,3:9,12:14)]
colnames(third_data)[1] <- "new_id"
write_xlsx(third_data,"3차가공데이터.xlsx")





```
