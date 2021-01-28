#Multiple Linear Regression

dataset=read.csv('50_Startups.csv')

dataset$State=factor(dataset$State,
                       levels=c('New York','California','Florida'),
                       labels=c(1,2,3))

library(caTools)
set.seed(123)
split=sample.split(dataset$Profit,SplitRatio=0.8)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#Fitting Multiple Linear Regression to training set
regressor=lm(formula= Profit ~ . , data=training_set)

#regressor=lm(formula= Profit ~ R.D.Spend , data=training_set)

#Predicting the test set result
Y_pred=predict(regressor,newdata=test_set)

