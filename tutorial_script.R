library(foreign)
library(caret)
library(car)
library(nlme)
library(rms)
library(e1071)
library(BiodiversityR)
library(moments)

data<-read.csv("~/statistical_analysis/qt51.csv")
names(data)
summary(data)

drop=c("subsystem","comp")

data = data[,!(names(data) %in% drop)]

withbug=subset(data, post_bugs>0)

withoutbug=subset(data, post_bugs==0)
mean(withoutbug$size)
mean(withbug$size)

summary(withoutbug$size)
summary(withbug$size)


plot(hist(withoutbug$size))
plot(density(withoutbug$size))

boxplot(withoutbug$size, data=withoutbug)
library(vioplot)
vioplot(withoutbug$size, withbug$size)


library(Hmisc)
describe(withoutbug$size)

library(psych)
describe(withoutbug$size)

drop=c("post_bugs")

independant=data[,!(names(data) %in% drop)]



##########correlation 
correlations <- cor(independant, method="spearman") 
highCorr <- findCorrelation(correlations, cutoff = .75)

low_cor_names=names(independant[, -highCorr])
low_cor_data= independant[(names(independant) %in% low_cor_names)]
dataforredun=low_cor_data

#############or varclus
vcobj = varclus ( ~., data = low_cor_data ,trans ="abs")
plot(vcobj)
abline(h=0.25, col="red")



#########start redun
redun_obj = redun (~. ,data = dataforredun ,nk =0)
after_redun= dataforredun[,!(names(dataforredun) %in% redun_obj $Out)]


############model
form=as.formula(paste("post_bugs>0~",paste(names(after_redun),collapse="+")))
model=glm(formula=form, data=log10(data+1), family = binomial(link = "logit"))
deviancepercentage(model)

newform=post_bugs>0~size+voter_ownership+little_discussion+median_discussion+ minor_wrote_no_major_revd+ major_actors+ median_minexp
newmodel=glm(formula=newform, data=log10(data+1), family = binomial(link = "logit"))


testdata=data.frame(size= log10(mean(data$size)+1), voter_ownership =log10(mean(data$voter_ownership)+1), little_discussion=log10(mean(data$little_discussion)+1), median_discussion=log10(mean(data$median_discussion)+1), minor_wrote_no_major_revd=log10(mean(data$minor_wrote_no_major_revd)+1), major_actors=log10(mean(data$major_actors)+1), median_minexp=log10(mean(data$median_minexp)+1))

predict(newmodel,testdata, type="response")

############new increase size

testdata=data.frame(size= log10(mean(data$size)*1.1+1), voter_ownership =log10(mean(data$voter_ownership)+1), little_discussion=log10(mean(data$little_discussion)+1), median_discussion=log10(mean(data$median_discussion)+1), minor_wrote_no_major_revd=log10(mean(data$minor_wrote_no_major_revd)+1), major_actors=log10(mean(data$major_actors)+1), median_minexp=log10(mean(data$median_minexp)+1))

predict(newmodel,testdata, type="response")

anova(newmodel)



#####################randomforest    classifier
library(randomForest)

rf.fit= randomForest(x=log10(after_redun+1), y=as.factor(data$post_bugs>0), ntree=100, type='classification', importance=TRUE)

predictions <- predict(rf.fit, data,type="response")
print(rf.fit)
TP = sum((predictions>0.5) & (data$post_bugs>0))
precision = TP / sum((predictions>0.5))
recall = TP / sum(data$post_bugs>0)
importance <- importance(rf.fit, type=1, class="TRUE",scale=FALSE)
importance <- as.data.frame(importance)
importance

