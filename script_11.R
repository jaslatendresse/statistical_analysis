library(foreign)
library(caret)
library(car)
library(nlme)
library(rms)
library(e1071)
library(BiodiversityR)
library(moments)
library(randomForest)

data<-read.csv("~/statistical_analysis/qt51.csv")
drop=c("subsystem","comp")

data = data[,!(names(data) %in% drop)]


k=10 #Folds

id <- sample(1:k,nrow(data),replace=TRUE)
list <- 1:k
trainingset <- data.frame()
testset <- data.frame()

fit=c();

precision=c();
recall=c();

precision_rf=c();
recall_rf=c();

for (i in 1:k)
{
  
  trainingset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))
  
  
  drop=c("post_bugs")
  
  independant=trainingset[,!(names(trainingset) %in% drop)]
  
  ##########correlation 
  correlations <- cor(independant, method="spearman") 
  highCorr <- findCorrelation(correlations, cutoff = .75)
  
  low_cor_names=names(independant[, -highCorr])
  low_cor_data= independant[(names(independant) %in% low_cor_names)]
  dataforredun=low_cor_data
  
  #########start redun
  redun_obj = redun (~. ,data = dataforredun ,nk =0)
  after_redun= dataforredun[,!(names(dataforredun) %in% redun_obj $Out)]
  
  
  
  
  ############model
  form=as.formula(paste("post_bugs>0~",paste(names(after_redun),collapse="+")))
  model=glm(formula=form, data=log10(trainingset+1), family = binomial(link = "logit"))
  
  predictions <- predict(model, log10(testset+1) ,type="response")
  TP = sum((predictions>0.5) & (testset$post_bugs>0))
  precision[i] = TP / sum((predictions>0.5))
  recall[i] = TP / sum(testset$post_bugs>0)
  
  fit[i]=1- model$deviance/model$null.deviance
  
  rf.fit= randomForest(x=log10(after_redun+1), y=as.factor(trainingset$post_bugs>0), ntree=100, type='classification', importance=TRUE)
  #rf.fit= randomForest(x=log10(independant+1), y=as.factor(data$AVERAGE_diff_Throughput), ntree=100, type='regression', importance=TRUE)
  
  rf_predictions <- predict(rf.fit, log10(testset+1),type="response")
  
  TP_rf = sum(rf_predictions=="TRUE" & (testset$post_bugs>0))
  precision_rf[i] = TP_rf / sum((rf_predictions=="TRUE"))
  recall_rf[i] = TP_rf / sum(testset$post_bugs>0)
  
}


#if without 10-fold

training_size= round(2*length(data$size)/3,digits = 0)
testing_size=length(data$size)-training_size
training_index=sample(nrow(data), training_size)

trainingset=data[training_index, ]
testset=data[-training_index, ]

