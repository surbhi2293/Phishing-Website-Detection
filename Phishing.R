library("randomForest")
library("e1071")
library("neuralnet")
library("boot")
lilbrary("plyr")

#Logistic Regression with simple CV

myData = Website_Phishing_Train
summary(myData)
dim(myData)
summary(myData$Result)
train = sample(2000, 1500)
lr.pred = rep(0,500)
lr.bands = glm(Result~., data=myData, subset=train, family = binomial)
lr.probs = predict(lr.bands, myData[-train,], type="response")
lr.pred[lr.probs>.5]=1
table(lr.pred,myData[-train,]$Result)
mean(lr.pred == myData[-train,]$Result)
 

#Random Forest with simple CV
myData_2 = myData
summary(myData_2$Result)
rf.model = randomForest(Result~., data=myData_2,subset=train,mtry=3,ntree=100,importance=TRUE)
rf.pred = predict(rf.model,myData_2[-train,],type="class")
importance(rf.model)
table(rf.pred,myData_2[-train,]$Result)
mean(rf.pred == myData_2[-train,]$Result)
 

#SVM with simple CV and cost=1
svm.lin = svm(Result~., data=myData_2[train,], kernel = "linear", cost=1, scale=FALSE)
svm.lin.pred = predict(svm.lin,myData_2[-train,])
table(svm.lin.pred,myData_2[-train,]$Result)
mean(svm.lin.pred == myData_2[-train,]$Result)
#Accuracy 
svm.lin = svm(Result~., data=myData_2[train,], kernel = "linear", cost=1, scale=TRUE)
table(svm.lin.pred,myData_2[-train,]$Result)
tune.lin = tune(svm,Result~., data=myData_2[-train,], kernel="linear", ranges=list(cost = c(.001, .01, .1, 1, 5, 10, 100)))
summary(tune.lin)
#Cost: 100
bestmod.lin = tune.lin$best.model
table(predict(bestmod.lin,myData_2[-train,]),myData_2[-train,]$Result)
mean(bestmod.lin == myData_2[-train,]$Result)

svm.nonlin_p = svm(Result~.,data=myData_2[train,],kernel="polynomial", gamma=1, cost=1)
table(predict(svm.nonlin_p,myData_2[-train,]),myData_2[-train,]$Result)
mean(svm.nonlin_p == myData_2[-train,]$Result)
#Accuracy 
svm.radial = svm(Result~.,data=myData_2[train,],kernel="radial", gamma=1, cost=1)
table(predict(svm.radial,myData_2[-train,]),myData_2[-train,]$Result)
mean(svm.radial == myData_2[-train,]$Result)

#LR scaled Dataset
summary(myData)
maxs = apply(myData, 2, max)
mins = apply(myData, 2, min)
myDataScaled = as.data.frame(scale(myData, center = mins, scale = maxs - mins))
index = sample(1:nrow(myDataScaled),round(0.75*nrow(myDataScaled)))
train = myDataScaled[index,]
test = myDataScaled[-index,]
lm.fit = glm(Result~.,data=train)
lm.pred = predict(lm.fit,test)
lm.MSE = sum((lm.pred-test$Result)^2)/nrow(test)
lm.MSE


#LR scaled CV - 10
lm.fit = glm(Result~., data=myDataScaled)
summary(lm.fit)
cv.glm(myDataScaled, lm.fit, K=10)$delta[1]
#

#neural net scaled
n = names(train)
n
f = as.formula(paste("Result ~", paste(n[!n %in% "Sex"], collapse = " + ")))
f
nn.fit = neuralnet(f,data=train,hidden=c(2),linear.output=TRUE)
plot(nn.fit)
dim(myData)
nn.pred = compute(nn.fit,test[,1:30])
names(nn.pred)
nn.MSE = sum((test$Result - nn.pred$net.result)^2)/nrow(test)
nn.MSE


#accuracy on unscaled data
nn.predUnscale = nn.pred$net.result*(max(myData$Result)-min(myData$Result))+min(myData$Result)
testMedvUnscale = (test$Result)*(max(myData$Result)-min(myData$Result))+min(myData$Result)
MSEUnscaled = sum((testMedvUnscale - nn.predUnscale)^2)/nrow(test)
sqrt(MSEUnscaled)
summary(myData$Result)
sqrt(MSEUnscaled)/mean(myData$Result)
# 0.93

pbar = create_progress_bar('text')
pbar$init(10)
for(i in 1:10){
  index = sample(1:nrow(myDataScaled),round(0.9*nrow(myDataScaled)))
  train.cv = myDataScaled[index,]
  test.cv = myDataScaled[-index,]
  nn.fit = neuralnet(f,data=train.cv,hidden=c(2),linear.output=TRUE,threshold=0.1)
  nn.pred = compute(nn.fit,test.cv[,2:9])
  cv.error[i] <- sum((test.cv$Result - nn.pred$net.result)^2)/nrow(test.cv)
  pbar$step()
}
mean(cv.error)
#0.242

# two hidden layers nN
nn.fit = neuralnet(f,data=train,hidden=c(4,2),linear.output=TRUE)
nn.pred = compute(nn.fit,test[,2:9])
nn.MSE_2 = sum((test$Result - nn.pred$net.result)^2)/nrow(test)
nn.MSE_2
#0.251

#SVM 10-fold CV
tuned = tune.svm(Result~., data = train, gamma = 10^-2, cost = 10^2, tunecontrol=tune.control(cross=10))
#Error 




