# install.packages(c("rgdal","dplyr","readxl","writexl","stringr","raster","geosphere","ape","Imap","igraph"))
library(rgdal)
library(ggplot2)
library(dplyr)
library(readxl)
library(writexl)
library(stringr)
library(raster)
library(geosphere)
library(igraph)
library(Imap)
library(ape)

map = readOGR("./EMD_202101/TL_SCCO_EMD.shp")
code_address = read_excel("./읍면동법적코드.xlsx")
map_list <- map@data
map_list[,"SIDO"] = as.numeric(substr(map_list$EMD_CD,start=1,stop= 2))


#읍면동 데이터만 남기기
df_code_address <- as.data.frame(code_address)
df_code_address$법정동코드<-floor(df_code_address$법정동코드/100)
df_code_address <- df_code_address[,-3]
names(df_code_address)[1] <- c("EMD_CD")
rm_data_ri <- grep("리$",df_code_address$법정동명, value = 1)
rm_data_gu <- grep("구$",df_code_address$법정동명,value = 1)
rm_data_si <- grep("시$",df_code_address$법정동명, value = 1)
rm_data_do <- grep("도$",df_code_address$법정동명, value = 1)
rm_data_gun <- grep("군$", df_code_address$법정동명, value = 1)
rm_data_special <- grep(")$", df_code_address$법정동명, value = 1)
df_code_address <- df_code_address[!(df_code_address$법정동명 %in% rm_data_si),]
df_code_address <- df_code_address[!(df_code_address$법정동명 %in% rm_data_do),]
df_code_address <- df_code_address[!(df_code_address$법정동명 %in% rm_data_gu),]
df_code_address <- df_code_address[!(df_code_address$법정동명 %in% rm_data_ri),]
df_code_address <- df_code_address[!(df_code_address$법정동명 %in% rm_data_gun),]
df_code_address <- df_code_address[!(df_code_address$법정동명 %in% rm_data_special),]
df_code_address$EMD_CD <- as.character(df_code_address$EMD_CD)
head(df_code_address)

# 읍면동 풀주소 데이터 합치기
join_data_EN <- left_join(df_code_address,map_list, by = "EMD_CD")


# join_data_EN 결측값 채워넣기
data_list <- c(1885,1943,2116,2469,4005,4493,4494,4495,4496,4497,4498,4499,4500)
data_name <- c()
data_sido <- c()
for(i in c(1:13)){
  address <- strsplit(join_data_EN$법정동명[data_list], split = " ")[[i]]
  while(length(address) > 1){
    address <- address[-1]
  }
  data_name <- c(data_name,address)
  data_sido <- floor(as.numeric(join_data_EN$EMD_CD[data_list])/1000000)
}

join_data_EN$SIDO[data_list] <- data_sido
join_data_EN$EMD_KOR_NM[data_list] <- data_name
join_data_EN$법정동명[1497] <- "세종특별자치시 산울동"
join_data_EN$법정동명[1498] <- "세종특별자치시 해밀동"
join_data_EN$법정동명[1499] <- "세종특별자치시 합강동"
join_data_EN$법정동명[1500] <- "세종특별자치시 집현동"
join_data_EN[is.na(join_data_EN),]
join_data_EN$EMD_ENG_NM[1885] <- "Baegot-dong"
join_data_EN$EMD_ENG_NM[1943] <- "Namsa-eup"
join_data_EN$EMD_ENG_NM[2116] <- "Saesol-dong"
join_data_EN$EMD_ENG_NM[2469] <- "Yeonggwimi-myeon"
join_data_EN$EMD_ENG_NM[4005] <- "Munmudaewang-myeon"
join_data_EN$EMD_ENG_NM[4493] <- "Yongji-dong"
join_data_EN$EMD_ENG_NM[4494] <- "Yongho-dong"
join_data_EN$EMD_ENG_NM[4495] <- "Sinwol-dong"
join_data_EN$EMD_ENG_NM[4496] <- "Daewon-dong"
join_data_EN$EMD_ENG_NM[4497] <- "Dudae-dong"
join_data_EN$EMD_ENG_NM[4498] <- "Samdong-dong"
join_data_EN$EMD_ENG_NM[4499] <- "Deokjeong-dong"
join_data_EN$EMD_ENG_NM[4500] <- "Toechon-dong"
join_data_EN[is.na(join_data_EN),]



# 공역 데이터 변수 추가
gongyuk <- read.csv("./공역-법정동.csv")
gongyuk$법정동 <- floor(gongyuk$법정동/100)
names(gongyuk)[3] <- c("EMD_CD")
gongyuk$EMD_CD <- as.character(gongyuk$EMD_CD)
join_data_EN_2 <- full_join(join_data_EN,gongyuk, by = "EMD_CD")
duplicated_data <- join_data_EN_2[duplicated(join_data_EN_2$EMD_CD),] # 중복된 데이터

### join_data_EN_2의 x.1 NA값 채워넣기 
load(file = "C:\\Users\\madda/Documents\\청년 데이터 캠프\\project\\UAM\\address_list.RData")
duplicated_data$법정동명 <- address_list
for(i in c(1:length(join_data_EN_2$법정동명))){
  if(is.na(join_data_EN_2$X.1)[i] == TRUE){
    join_data_EN_2$X.1[i] <- join_data_EN_2$법정동명[i]
  }
}
duplicated_data <- duplicated_data[,c(2,8)]


### 변수 비행불가능여부 NA값 채워넣기
airspace <- inner_join(join_data_EN_2,duplicated_data, by = "법정동명")
airspace <- airspace[,-c(6,8)]
duplicated_airspace <- airspace[duplicated(airspace$EMD_CD),] # airspace의 중복으로 있는 데이터
airspace <- airspace[-as.numeric(rownames(duplicated_airspace)),]
join_data_EN_2 <- join_data_EN_2[-as.numeric(rownames(duplicated_data)),]
join_data_EN_2[join_data_EN_2$법정동명 %in% airspace$법정동명,]$비행불가능여부 <- airspace$비행불가능여부.y
join_data_EN_2[is.na(join_data_EN_2$비행불가능여부),] 
join_data_EN_3 <- join_data_EN_2[,-c(6,7)]
write_xlsx(join_data_EN_3,"join_data_EN_3.xlsx")
### 여기 비행불가능여부 na값 채워넣기


# 터미널유무 변수 추가
bus_station <- read.csv("./터미널법정동.csv")
bus_station_copy <- bus_station[,-c(1,2,3)]
bus_station_copy$코드 <- floor(bus_station_copy$코드/100)
names(bus_station_copy)[1] <- "EMD_CD"
bus_station_copy$EMD_CD <- as.character(bus_station_copy$EMD_CD)
bus_join_data <- left_join(join_data_EN_3, bus_station_copy, by ="EMD_CD")
write_xlsx(bus_join_data,"bus_join_data.xlsx")
bus_join_data <- read_excel("./bus_join_data.xlsx") %>% as.data.frame()
bus_join_data<-bus_join_data[!(duplicated(bus_join_data$법정동명)),]
bus_join_data$터미널유무[is.na(bus_join_data$터미널유무)] <- 0 
write_xlsx(bus_join_data,"공항여부_터미널여부_미완성.xlsx")

# 교통혼잡,인구수 데이터 변수 추가
trans <- read_excel("./교통관련데이터.xlsx") %>% as.data.frame()
trans_copy <- trans[,-10]
trans_copy$행정주소 <- paste(trans_copy$`도/광역시/시`,trans_copy$시군구,trans_copy$읍면동,
                         rep = " ")
trans_copy <- trans_copy[,-c(1,2,3)]
trans_copy <- trans_copy[,c(7,1:6)]
trans_copy
trans_copy[is.na(trans_copy),]
write_xlsx(trans_copy,"trans_copy.xlsx")

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
sum(duplicated(copy_relation_data))
write_xlsx(copy_relation_data,"copy_relation_data.xlsx")

## copy_relation_data 결측값 수정
copy_relation_data$행정구역코드[copy_relation_data$법정동 == "신수동" & copy_relation_data$시군구 == "사천시"] <- 0
copy2_relation_data <- na.omit(copy_relation_data) %>% as.data.frame()
sum(is.na(copy2_relation_data))


# bus_join_data, copy2_relation_data 합치기
str(bus_join_data)
str(copy2_relation_data)
bus_join_data$EMD_CD <- as.numeric(bus_join_data$EMD_CD)
new_final_data <- left_join(copy2_relation_data,bus_join_data, by = "EMD_CD")
str(new_final_data)
sum(duplicated(new_final_data))
new_final_data <- new_final_data[,-10]
names(new_final_data)[10] <- "법정동명"
write_xlsx(new_final_data,"./new_final_data.xlsx")


# trans_copy, new_final_data 합치기
trans_copy$행정주소
trans_copy$행정주소 <- str_trim(trans_copy$행정주소, side = "right")
trans_copy$행정주소
new_final_data <- read_excel("./new_final_data.xlsx")
new_final_data_2 <- left_join(new_final_data,trans_copy, by ="행정주소")
sum(is.na(new_final_data_2))
sum(is.na(new_final_data_2$비행불가능여부))
sum(is.na(new_final_data_2$승용차))
first_data_ver2 <- new_final_data_3[,c(8,12,13,9,10,14:21)]
names(first_data_ver2)[5] <- "법정동주소"
write_xlsx(first_data_ver2,"first_data_ver2.xlsx")
## 여기서 수작업으로 행정주소와 법정동주소가 일치하지 않는 것은 엑셀에서 수정


#2차가공데이터 전처리
second_data <- read_excel("./2차가공데이터_ver2.xlsx")
sum(duplicated(second_data$EMD_CD))
sum(is.na(second_data$승용차))
sum(is.na(second_data$비행불가능여부))
sum(is.na(second_data))
View(second_data[duplicated(second_data$EMD_CD),])
third_data <- second_data[!duplicated(second_data$EMD_CD),]
sum(duplicated(third_data))
sum(is.na(third_data))
sum(is.na(third_data$비행불가능여부))
sum(is.na(third_data$승용차))

## 아예 관측을 안한 데이터 제거
third_copy_data <- third_data[third_data$행정주소 == third_data$법정동주소,]
no_reserve_data_list <-third_copy_data[is.na(third_copy_data$승용차),]
third_propressed_data <- third_data[!(third_data$EMD_CD %in% no_reserve_data_list$EMD_CD),]

# 행정주소와 법정동주소 같지 않은 데이터 제거
third_no_na_data <- third_data[!is.na(third_data$승용차),]
write_xlsx(third_no_na_data,"3차가공데이터.xlsx")
# 3차가공데이터에 비행불가능여부지역na값과 기차역변수 추가해준다.



# 3차가공데이터의 na값을 추가한 값과, 기차역추가한 변수 불러오기
data <- read_excel("./4차가공데이터.xlsx")


# 읍면동 구분을 위해 id변수 만든 후 데이터 합치기
map_list <- map@data
map_list[,"id"] <- (1:nrow(map_list)) - 1
map_list <- map_list[,c(1,4)]
map_list$EMD_CD <- as.numeric(map_list$EMD_CD)
fusion_data <- left_join(data,map_list, by = "EMD_CD")



# Max-Min 정규화함수 만들기
mm_normal <- function(x){
  (x - min(x)) / (max(x) - min(x))
}


mm_normal_data <- fusion_data
for(i in c(9,10,11,13,14)){
  mm_normal_data[i] <- mm_normal(mm_normal_data[i])
}

# 시각화 할 데이터 만들기
## 원래 알던 위/경도 값으로 전환
map <- spTransform(map, CRS("+proj=longlat")) 
korea <- fortify(map)
korea

# 위/경도값 구하기
long_check_max <- c()
long_check_min <- c()
lat_check_max <- c()
lat_check_min <- c()
for(i in c(0:5050)){
  long_check_max <- c(long_check_max,max(korea$long[korea$id == i]))
  long_check_min <- c(long_check_min,min(korea$long[korea$id == i]))
  lat_check_max <- c(lat_check_max,max(korea$lat[korea$id == i]))
  lat_check_min <- c(lat_check_min,min(korea$lat[korea$id == i]))
  print(i)
}

long <- data.frame(long_check_max,long_check_min)
lat <- data.frame(lat_check_max,lat_check_min)

long$long <- apply(long,1,mean)
lat$lat <- apply(lat,1,mean)

location <- data.frame(long$long,lat$lat)
location$id <- c(0:5050)

fiveth_data <- inner_join(fusion_data,location, by ="id")
write_xlsx(fiveth_data,"5차가공데이터.xlsx")

# 읍면동 중심위치와 가까운 터미널 거리 구하기
terminal <- read_excel("./터미널좌표.xlsx")
terminal <- terminal[,-1]
terminal_dist <- c()
for(i in c(1:length(fiveth_data$long.long))){
  data <- c()
  data <- gdist(lon.1 = fiveth_data$long.long[i],
                lat.1 = fiveth_data$lat.lat[i],
                lon.2 = terminal$위도,
                lat.2 = terminal$경도,
                units = "km")
  terminal_dist <- c(terminal_dist, min(data))
}
terminal_dist


# 읍면동 중심위치와 가까운 기차역 거리 구하기
train_station_data <- read_excel("./기차역좌표.xlsx")
train_station_data <- train_station_data[,-1]
train_station_dist <- c()

for(i in c(1:length(fiveth_data$long.long))){
  train_data <- c()
  train_data <- gdist(lon.1 = fiveth_data$long.long[i],
                      lat.1 = fiveth_data$lat.lat[i],
                      lon.2 = train_station_data$위도,
                      lat.2 = train_station_data$경도,
                      units = "km")
  train_station_dist <- c(train_station_dist,min(train_data))
}


# 읍면동 중심위치와 가까운 공항 거리 구하기
airport_data <- read.csv("./공항위치.csv")
colnames(airport_data) <- c("공항","경도","위도")
airport_data <- airport_data[,-1]
airport_dist <- c()

for(i in c(1:length(fiveth_data$long.long))){
  airport <- c()
  airport <- data <- gdist(lon.1 = fiveth_data$long.long[i],
                           lat.1 = fiveth_data$lat.lat[i],
                           lon.2 = airport_data$위도,
                           lat.2 = airport_data$경도,
                           units = "km")
  airport_dist <- c(airport_dist,min(airport))
}

# 3개 거리 합치기
distance <- data.frame(terminal_dist,train_station_dist,airport_dist)
colnames(distance) <- c("터미널과의거리","기차역과의거리","공항과의거리")
write_xlsx(distance,"교통수단(3개)과의거리.xlsx")

# 6차 데이터 만들기 : 5차가공데이터와 합치기기
sixth_data <- data.frame(fiveth_data,distance)
names(sixth_data)[17] <- "Lon"
names(sixth_data)[18] <- "Lat"
write_xlsx(sixth_data,"6차가공데이터.xlsx")



# 7차 데이터 만들기(안쓰는 변수 제거)
sixth_data <- read_excel("./6차가공데이터.xlsx")
seventh_data <- data.frame(c(0:4915),sixth_data[,c(5,17,18,9:11,13,19:21)])
names(seventh_data)[1] <- "id"
write_xlsx(seventh_data,"7차가공데이터.xlsx")


seventh_data[3945,]
seventh_data <- seventh_data[-3945,]
