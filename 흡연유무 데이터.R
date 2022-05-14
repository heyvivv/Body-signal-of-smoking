getwd()
setwd('/Users/leehyebin/Rdata')

smoking <- read.csv(file = 'smoking.csv', stringsAsFactors = TRUE)
head(smoking)
df <- smoking[, -1]
head(df)
str(df)
df$smoking <- as.factor(df$smoking)
df$dental.caries <- as.factor(df$dental.caries)

library(caret)
set.seed(1234)
idx <- createDataPartition(df$smoking, p = 0.1)
df_2 <- df[idx$Resample1, ]
str(df_2)
table(df$smoking)
table(df_2$smoking)

summary(df_2)
dim(df_2)
str(df_2)
sum(is.na(df))

#탐색적 분석으로 첨도 및 왜도 확인
options(digits = 3)
library(pastecs)
library(psych)
describeBy(df_2[, -c(1, 23, 24, 25, 26)], df_2$smoking, mat = FALSE)

##상관관계 확인
df_numeric <- df_2[, -c(1, 23, 24, 25, 26)]
cor_df <- cor(df_numeric)
cor_df
library(PerformanceAnalytics)
chart.Correlation(cor_df, histogram = TRUE, pch = 19)
#상관관계가 0.8이상인 변수들이 눈에 보인다. 제거해주자.
library(caret)
findCorrelation(cor_df, cutoff = 0.75)
str(df_numeric)
new_df <- df_numeric[, -c(3, 9)]
str(new_df)

#분산이 0에 가까운 변수도 제거한다.
nearZeroVar(new_df, saveMetrics = TRUE)
new_df2 <- new_df[, -nearZeroVar(new_df)]
str(new_df2)

#팩터형 변수들을 다시 합친다
str(df_2)
new_df2$gender <- df_2$gender
new_df2$dental.caries <- df_2$dental.caries
new_df2$tartar <- df_2$tartar
new_df2$smoking <- df_2$smoking
str(new_df2)

#설명변수가 여전히 19개로 다소 많다. 카이제곱 검정을 사용한 독립성 검정으로 변수의 가중치를 계산한다.
library(FSelector)
cs <- chi.squared(smoking ~., data = new_df2)
cutoff.k(cs, 12)
select <- cutoff.k(cs, 12)
df <- new_df2[, select]
str(df)
df$smoking <- new_df2$smoking
str(df)
#설명변수를 12개로 줄였다.

#박스플롯 확인 
boxplot(df[, c(1, 2)])
boxplot(df[, c(3, 4)])

outHigh <- function(x) {
  x[x > quantile(x, 0.95)] <- quantile(x, 0.75)
  x
}

outLow <- function(x) {
  x[x < quantile(x, 0.05)] <- quantile(x, 0.25)
  x
}

str(df)
df_2 <- data.frame(lapply(df[, -c(1, 11, 13)], outHigh))
df_2 <- data.frame(lapply(df_2, outLow))

df_2$gender <- df$gender
df_2$tartar <- df$tartar
df_2$smoking <- df$smoking

str(df_2)
boxplot(df_2[, c(1, 2)])
boxplot(df_2[, c(3, 4)])

#데이터 분리
library(caret)
idx <- createDataPartition(df_2$smoking, p = 0.7)
train <- df_2[idx$Resample1, ]
test <- df_2[-idx$Resample1, ]

table(train$smoking) #결과변수의 데이터 수가 다소 차이나긴 하나 큰 차이는 아니므로 업샘플링 하지 않는다

#데이터 정규화가 필요한 모델링을 위해 미리 정규화 작업
model_train <- preProcess(train, method = c('center', 'scale'))
model_test <- preProcess(test, method = c('center', 'scale'))

scaled_train <- predict(model_train, train)
scaled_test <- predict(model_test, test)





#SVM : 정규화 데이터 사용
library(class)
library(kknn)
library(e1071)
library(caret)
library(MASS)
library(reshape2)
library(ggplot2)
library(kernlab)
library(corrplot)

linear.tune <- tune.svm(smoking ~., data = scaled_train,
                        kernel = 'linear',
                        cost = c(0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10))
summary(linear.tune)
#최적의 cost함수는 0.1로 나왔고, 분류 오류 비율은 대략 25.7% 정도이다.

#predict()함수로 test 데이터로 예측을 실행해보자.
best.linear <- linear.tune$best.model
tune.test <- predict(best.linear, newdata = scaled_test)
confusionMatrix(tune.test, scaled_test$smoking, positive = '1')
#정확도는 약 72%이고 카파 통계량은 0.41로 나왔다. 비선형 방법을 이용해 성능을 조금 더 높혀보자
#커널함수로 polynomial을 사용, 차수는 3, 4, 5의 값을 주고 커널계수는 0.1부터 4까지의 숫자를 준다
set.seed(123)
poly.tune <- tune.svm(smoking ~., data = scaled_train,
                      kernel = 'polynomial', degree = c(3, 4, 5),
                      coef0 = c(0.1, 0.5, 1, 2, 3, 4))
summary(poly.tune)
#이 모형은 다항식의 차수 degree의 값으로 3, 커널계수는 0.5를 선택했다. 

#predict()함수로 test데이터 예측을 실행해보자
best.poly <- poly.tune$best.model
poly.test <- predict(best.poly, newdata = scaled_test)
confusionMatrix(poly.test, scaled_test$smoking, positive = '1')
#정확도는 약 73.4% 카파통계량은 0.445로 선형모형보다 약간 더 성능이 좋아졌다

#다음으로 방사 기저 함수(radial basis function)을 사용해보자.
#매개변수로 gamma를 사용하고 최적값을 찾기위해 0.1부터 4까지 증가시켜 본다.
#gamma값이 너무 작을 때는 모형이 결정 분계선을 제대로 포착하지 못할 수 있고, 
#값이 너무 클때는 모형이 지나치게 과적합될 수 있으므로 주의가 필요하다.
set.seed(123)
rbf.tune <- tune.svm(smoking ~., data = scaled_train,
                     kernel = 'radial', 
                     gamma = c(0.1, 0.5, 1, 2, 3, 4))
summary(rbf.tune)
#최적의 gamma값으로 0.1이 나왔다. predict()함수로 성능을 보자
best.rbf <- rbf.tune$best.model
rbf.test <- predict(best.rbf, newdata = scaled_test)
confusionMatrix(rbf.test, scaled_test$smoking, positive = '1')
#정확도 74.1%, 카파통계량 0.456으로 성능이 약간 더 좋아졌다.

#마지막으로 커널함수를 시그모이드 함수로 설정해보자 
set.seed(123)
sigmoid.tune <- tune.svm(smoking ~., data = scaled_train, 
                         kernel = 'sigmoid',
                         gamma = c(0.1, 0.5, 1, 2, 3, 4),
                         coef0 = c(0.1, 0.5, 1, 2, 3, 4))
best.sigmoid <- sigmoid.tune$best.model
sigmoid.test <- predict(best.sigmoid, newdata = scaled_test)
confusionMatrix(sigmoid.test, scaled_test$smoking, positive = '1')
#정확도 65.9%, 카파통계량 0.268로 결과가 매우 좋지 않다. 
#SVM 모형 중 방사기저함수 커널을 사용한 모형이 정확도 74.1%, 카파통계량 0.456으로 가장 좋은 성능을 보였다.






#랜덤포레스트를 이용한 분류
library(rpart)
library(partykit)
library(MASS)
library(genridge)
library(randomForest)
library(xgboost)
library(caret)
library(Boruta)

set.seed(234)
rf.biop <- randomForest(smoking ~., data = train)
rf.biop
#수행결과 OOB 오차율은 24.6%로 나온다. 
#개선을 위해 최적의 트리수를 보자.
plot(rf.biop)
which.min(rf.biop$err.rate[, 1])
#모형 정확도를 최적화하기에 필요한 트리수가 338개면 된다는 결과를 얻었다. 
set.seed(234)
rf.biop2 <- randomForest(smoking ~., data = train, ntree = 338)
print(rf.biop2)
#OOB오차가 24.6%에서 24.2%로 아주 약간 개선되었다. 
#test데이터로 어떤 결과가 나오는지도 보자.
rf.biop.test <- predict(rf.biop2, newdata = test, type = 'response')
confusionMatrix(rf.biop.test, test$smoking, positive = '1')
#약 74.9%의 정확도, 카파통계량 0.47로 SVM 모형 중 방사기저함수 커널을 사용한 모형(정확도 74.1%, 카파통계량 0.456)보다 더 개선되었다.





#앙상블 분석
library(caret)
library(MASS)
library(caretEnsemble)
library(caTools)
library(smotefamily)
library(mlr)
library(ggplot2)
library(HDclassif)
library(reshape2)
library(corrplot)

levels(train$smoking)[levels(train$smoking) == 0] <- 'No'
levels(train$smoking)[levels(train$smoking) == 1] <- 'Yes'
levels(test$smoking)[levels(test$smoking) == 0] <- 'No'
levels(test$smoking)[levels(test$smoking) == 1] <- 'Yes'

control <- trainControl(method = 'cv', number = 5, 
                        savePredictions = 'final',
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
set.seed(123)
models <- caretList(smoking ~., data = train,
                    trControl = control,
                    metric = 'ROC',
                    methodList = c('rpart', 'earth', 'knn'))
models
modelCor(resamples(models))
#rpart와 earth간의 상관관계가 0.92로 높다. 따라서 둘 중 한 모델을 빼고 다른 모델로 대체한다.

set.seed(123)
models2 <- caretList(smoking ~., data = train,
                    trControl = control,
                    metric = 'ROC',
                    methodList = c('treebag', 'earth', 'nnet'))
models2
modelCor(resamples(models2))
#stack으로 로지스틱 회귀모형을 쌓아보자
model_preds <- lapply(models2, predict, newdata = test, type = 'prob')
model_preds <- lapply(model_preds, function(x) x[, 'Yes'])
model_preds <- data.frame(model_preds)
#caretStack함수를 이용해 모형을 쌓는다. 
stack <- caretStack(models2, method = 'glm', metric = 'ROC', 
                    trControl = trainControl(
                      method = 'boot', number = 5, 
                      savePredictions = 'final',
                      classProbs = TRUE,
                      summaryFunction = twoClassSummary
                    ))
summary(stack)
#모형들의 계수가 다 유의하다. 
#앙상블에 사용된 학습자의 각 예측결과를 비교해보자.
prob <- 1 - predict(stack, newdata = test, type = 'prob')
model_preds$ensemble <- prob
colAUC(model_preds, test$smoking)
#각 모형의 AUC를 봤을 때 앙상블 모형이 earth만 사용한 것보다 좀 더 나은 성능을 보였다. 
