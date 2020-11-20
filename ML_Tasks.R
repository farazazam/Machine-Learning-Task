library(mlr)

library(tidyverse)
library(data.table)
raw <- readxl::read_xlsx("C:/Users/SyedFar/Desktop/google.xlsx",sheet = "Sheet1")

model_configs = read.xlsx("C:/Users/SyedFar/Desktop/google.xlsx",sheet = "Sheet2")

 
raw1<- raw[,model_configs$variable]

raw1[raw1==0] <- NA

raw2 <- as.data.table(raw1)

googleClean <- mutate_all(raw2,as.numeric)%>%
  filter(is.na(DMA_Google_Queries_Index_Adj)==FALSE)

imputeMethod <- imputeLearner("regr.rpart")
googleImp <- impute(as.data.frame(googleClean), classes = list(numeric= imputeMethod))
googleImp1<-googleImp$data
bad <- sapply(googleImp1, function(x) all(is.nan(x)))
googleImp2 <- googleImp1[,!bad]
#Defining our task and learner

n= nrow(googleImp2)
googleImp2_train<- googleImp2[seq(1,n,by=2),]
googleImp2_test <- googleImp2[seq(2,n,by=2),]


googleTask <- makeRegrTask(data= googleImp2_train, target = "DMA_Google_Queries_Index_Adj")

lin <- makeLearner("regr.lm")

#Using a filter menthod for feature selection

filterVals <- generateFilterValuesData(googleTask, method = "linear.correlation")


#Manually selecting which features to drop

googleFiltTask <- filterFeatures(googleTask, fval = filterVals, abs = 44)

googleFiltTask <- filterFeatures(googleTask, fval = filterVals, perc = 0.03744)

googleFiltTask <- filterFeatures(googleTask, fval = filterVals, threshold = 0.3)


#Creating a filter wrapper

filterWrapper = makeFilterWrapper(learner = lin, fw.method = "linear.correlation")

#Tuning the number of predictors to retain
lmParamSpace <- makeParamSet(makeIntegerParam("fw.abs", lower = 1, upper = 44))


gridSearch <- makeTuneControlGrid()

kFold <- makeResampleDesc("CV", iters=10)
tunedFeats <- tuneParams(filterWrapper, task = googleTask, resampling = kFold, par.set = lmParamSpace, control = gridSearch)


#Training the model with filtered features

filteredTask <- filterFeatures(googleTask,fval = filterVals, abs = unlist(tunedFeats$x))

filteredModel <- train(lin, filteredTask)


wrapperModelData <- getLearnerModel(filteredModel)
sink("lm.txt")
print(summary(wrapperModelData))
sink()


featSelControl <- makeFeatSelControlSequential(method = "sfbs")
selFeats <- selectFeatures(learner = lin, task = googleTask, resampling = kFold, control = featSelControl)

googleSelFeat <- googleImp2[,c("DMA_Google_Queries_Index_Adj",selFeats$x)]
googleSelFeatTask <- makeRegrTask(data = googleSelFeat, target = "DMA_Google_Queries_Index_Adj")
wrapperModel <- train(lin,googleSelFeatTask)

wrapperModelData <- getLearnerModel(wrapperModel)
sink("lm.txt")
print(summary(wrapperModelData))
sink()



################Non Linear Models###############



gamTask<- makeRegrTask(data=googleImp2, target = "DMA_Google_Queries_Index_Adj")
imputeMethod <- imputeLearner("regr.rpart")
gamImputeWrapper <- makeImputeWrapper("regr.gamboost", classes = list(numeric= imputeMethod))
gamFeatSelControl <- makeFeatSelControlSequential(method = "sfbs")
kFold <- makeResampleDesc("CV", iters=3)
gamFeatSelWrapper <- makeFeatSelWrapper(learner= gamImputeWrapper, resampling = kFold,control = gamFeatSelControl)


holdout <- makeResampleDesc("Holdout")
gamCV<- resample(gamTask, resampling = holdout)

###Model####

lin1 <- makeLearner("regr.gamboost")
selFeats <- selectFeatures(learner = lin1, task = googleTask, resampling = kFold, control = featSelControl)


googleSelFeat <- googleImp2[,c("DMA_Google_Queries_Index_Adj",selFeats$x)]
googleSelFeatTask <- makeRegrTask(data = googleSelFeat, target = "DMA_Google_Queries_Index_Adj")
wrapperModel <- train(lin1,googleSelFeatTask)



wrapperModelData <- getLearnerModel(wrapperModel,)
sink("lm4.txt")
print(summary(wrapperModelData))
sink()


par(mfrow = c(3, 3))

plot(wrapperModelData, type = "l")

plot(wrapperModelData$fitted(), resid(wrapperModelData))

qqnorm(resid(wrapperModelData))

qqline(resid(wrapperModelData))

par(mfrow = c(1, 1))

x <-wrapperModelData$fitted()






googleImp3$DMA_Google_Queries_Index_Adj<- log2(googleImp3$DMA_Google_Queries_Index_Adj)
#Building your first Ridge, Lasso, and elastic net models
googleImp3 <- googleImp2_test
googleImp3<-as.data.table(googleImp3)
googleTask <- makeRegrTask(data= googleImp3, target = "DMA_Google_Queries_Index_Adj")


ridge <- makeLearner("regr.glmnet", alpha = 0, id = "ridge")


ridgeParamSpace <- makeParamSet(makeNumericParam("s", lower = 0, upper = 100))
randSearch <- makeTuneControlRandom(maxit = 200)
cvForTuning <- makeResampleDesc("RepCV", folds=3, reps=10)

library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())
tunedRidgePars <- tuneParams(ridge, task = googleTask, resampling = cvForTuning, par.set = ridgeParamSpace, control = randSearch)
parallelStop()


ridgeTuningData <- generateHyperParsEffectData(tunedRidgePars)
plotHyperParsEffect(ridgeTuningData, x= "s", y= "mse.test.mean", plot.type = "line")+theme_bw()


tunedRidge <- setHyperPars(ridge, par.vals = tunedRidgePars$x)
tunedRidgeModel <- train(tunedRidge,googleTask)


ridgeModelData <- getLearnerModel(tunedRidgeModel,more.unwrap = TRUE)
ridgeCoefs <- coef(ridgeModelData, s= tunedRidgePars$x$s)


#Training the lasso model
lasso <- makeLearner("regr.glmnet", alpha= 1, id= "lasso")
lassoParamSpace <- makeParamSet(makeNumericParam("s", lower = 0, upper = 100))
parallelStartSocket(cpus = detectCores())
tunedLassoPars <- tuneParams(lasso, task = googleTask, resampling = cvForTuning,par.set = lassoParamSpace, control = randSearch)
parallelStop()
tunedLassoPars

lassoTuningData <- generateHyperParsEffectData(tunedLassoPars)
plotHyperParsEffect(lassoTuningData, x = "s", y = "mse.test.mean", plot.type = "line")+theme_bw()

tunedLasso <- setHyperPars(lasso, par.vals = tunedLassoPars$x)
tunedLassoModel <- train(tunedLasso, googleTask)

lassoModelData <- getLearnerModel(tunedLassoModel)
lassoCoefs <- coef(lassoModelData, s= tunedLassoPars$x$s)
lassoCoefs

#Training the elastic net model

elastic <- makeLearner("regr.glmnet", id= "elastic")
elasticParamSpace <- makeParamSet(makeNumericParam("s", lower = 0, upper = 100), makeNumericParam("alpha", lower = 0, upper = 1))
randSearchElastic <- makeTuneControlRandom(maxit = 400)
parallelStartSocket(cpus = detectCores())
tunedElasticPars <- tuneParams(elastic, task = googleTask, resampling = cvForTuning, par.set = elasticParamSpace, control = randSearchElastic)
parallelStop()
tunedElasticPars

tunedElastic <- setHyperPars(elastic, par.vals = tunedElasticPars$x)
tunedElasticModel <- train(tunedElastic, googleTask)
elasticModelData <- getLearnerModel(tunedElasticModel,more.unwrap = TRUE)
elasticCoefs <- coef(elasticModelData, s= tunedElasticPars$x$s)

sink("lm3.txt")
print(summary(elasticCoefs))
sink()







###Random Forest Model
forest <- makeLearner("regr.randomForest")
kFold <- makeResampleDesc("CV", iters = 10)
forestParamSpace <- makeParamSet(makeIntegerParam("ntree", lower = 50, upper = 50),
                                 makeIntegerParam("mtry", lower = 20, upper = 103),
                                 makeIntegerParam("nodesize", lower = 1, upper = 10),
                                 makeIntegerParam("maxnodes", lower = 5,upper = 30))
randSearch <- makeTuneControlRandom(maxit = 100)
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task= googleTask, resampling = kFold, par.set = forestParamSpace, control = randSearch)

parallelStop()
tunedForestPars


tunedForest <- setHyperPars(forest, par.vals = tunedForestPars$x)
tunedForestModel <- train(tunedForest, googleTask)
forestModelData <- getLearnerModel(tunedForestModel)

plot(forestModelData)



##XGboost Model

xgb <- makeLearner("regr.xgboost")

xgbParamSpace <- makeParamSet(makeNumericParam("eta", lower = 0, upper = 1),
                              makeNumericParam("gamma", lower = 0, upper = 10),
                              makeIntegerParam("max_depth", lower = 1, upper = 20),
                              makeNumericParam("min_child_weight", lower = 1, upper = 10),
                              makeNumericParam("subsample", lower = 0.5, upper = 1),
                              makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
                              makeIntegerParam("nrounds", lower = 30, upper = 30))

tunedXgbPars <- tuneParams(xgb, task = googleTask, resampling = kFold, par.set = xgbParamSpace, control = randSearch)
tunedXgbPars



tunedXgb <- setHyperPars(xgb, par.vals = tunedXgbPars$x)
tunedXgbModel <- train(tunedXgb, googleTask)
xgbModelData <- getLearnerModel(tunedXgbModel)
ggplot(xgbModelData$evaluation_log, aes(iter, train_rmse)) +
  geom_line() +
  geom_point() +
  theme_bw()


outer <- makeResampleDesc("CV", iters = 6)
cvForTuning <- makeResampleDesc("CV", iters = 5)
xgbWrapper <- makeTuneWrapper("regr.xgboost",
                              resampling = cvForTuning,
                              par.set = xgbParamSpace,
                              control = randSearch)

cvWithTuning <- resample(xgbWrapper, googleTask, resampling = outer)
cvWithTuning

x<-getLearnerModel(cvWithTuning)
sink("lm7.txt")
print(summary(cvWithTuning))
sink()



#####Predictions


####Ridge
test<-as.data.table(googleImp2_test)
newdata.pred=predict(tunedRidgeModel, newdata= test)
newdata.pred



####Linear Regression



newdata.pred=predict(filteredModel, newdata= googleImp2_test)
newdata.pred



#####XG Boost

cvWithTuning

newdata.pred=predict(tunedXgbModel, newdata= googleImp2_test)
newdata.pred


#####Random Forest

newdata.pred=predict(tunedForestModel, newdata= googleImp2_test)
newdata.pred



#######Elastic Nets
test$DMA_Google_Queries_Index_Adj<- log2(test$DMA_Google_Queries_Index_Adj)
test<- as.data.table(googleImp2_test)

newdata.pred=predict(tunedElasticModel, newdata= test)
newdata.pred



##XGboost Model Second trail by including interaction terms and square terms

googleImp2 <- googleImp1[,!bad]

####Squaring terms

squaredgoogleImp2 <- googleImp2^2

#Defining our task and learner

n= nrow(googleImp2)
googleImp2_train<- googleImp2[seq(1,n,by=2),]
googleImp2_test <- googleImp2[seq(2,n,by=2),]










xgb <- makeLearner("regr.xgboost")

xgbParamSpace <- makeParamSet(makeNumericParam("eta", lower = 0, upper = 1),
                              makeNumericParam("gamma", lower = 0, upper = 10),
                              makeIntegerParam("max_depth", lower = 1, upper = 20),
                              makeNumericParam("min_child_weight", lower = 1, upper = 10),
                              makeNumericParam("subsample", lower = 0.5, upper = 1),
                              makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
                              makeIntegerParam("nrounds", lower = 30, upper = 30))

tunedXgbPars <- tuneParams(xgb, task = googleTask, resampling = kFold, par.set = xgbParamSpace, control = randSearch)
tunedXgbPars



tunedXgb <- setHyperPars(xgb, par.vals = tunedXgbPars$x)
tunedXgbModel <- train(tunedXgb, googleTask)
xgbModelData <- getLearnerModel(tunedXgbModel)
ggplot(xgbModelData$evaluation_log, aes(iter, train_rmse)) +
  geom_line() +
  geom_point() +
  theme_bw()


outer <- makeResampleDesc("CV", iters = 6)
cvForTuning <- makeResampleDesc("CV", iters = 5)
xgbWrapper <- makeTuneWrapper("regr.xgboost",
                              resampling = cvForTuning,
                              par.set = xgbParamSpace,
                              control = randSearch)

cvWithTuning <- resample(xgbWrapper, googleTask, resampling = outer)
cvWithTuning





##########Projected Leads##############



library(mlr)

library(tidyverse)
library(data.table)


theData<-  read.csv("C:/Users/SyedFar/Desktop/adstock.csv", header=TRUE, sep=",")
adstock<-function(x, r) {
  tempX<-rep(NA,length(x))
  x[is.na(x)]<-0
  for(i in 1:length(x)){
    if (i==1) tempX[i] <- x[i]*r else tempX[i] <- x[i]*r + (1-r) * tempX[i-1]
  }
  return (tempX)
}


cnames <- names(theData)
cnames <- cnames[-1]

mAdstock1<-function(df, var, r=c(.1,.2,.3,.4,.5,.6,.7,.8,.9)) {
  for (v in var) {
    for (i in r){
      df[,paste0(v,i*100)] <- adstock(df[,v], i)
    }
  }
  return(df)
}

adtocked_df <- mAdstock1(theData, cnames)



Projected_leads <- readxl::read_xlsx("C:/Users/SyedFar/Desktop/ML.Task/Raw-Data-DMA_Projected_Visits_MA3-2020-11-18.xlsx",sheet = "Sheet1")



lockdown <- read.csv("C:/Users/SyedFar/Desktop/HVA Model_11_6_2020/Overall_Visits/lockdown.csv")








raw2$WeeklyData


raw <- readxl::read_xlsx("C:/Users/SyedFar/Desktop/ML.Task/Raw-Data-DMA_Projected_Visits_MA3-2020-11-18.xlsx",sheet = "Sheet2")



raw[raw==0] <- NA

raw2 <- as.data.table(raw)

Projected_leadsclean <- raw2%>%mutate_at(vars(-one_of("WeeklyData","DMA_Projected_Visits_MA3")),as.numeric)
  

imputeMethod <- imputeLearner("regr.rpart")
ProjectedLeadsImp <- impute(as.data.frame(Projected_leadsclean[,-1]), classes = list(numeric= imputeMethod))
ProjectedLeadsImp1<-ProjectedLeadsImp$data
bad <- sapply(ProjectedLeadsImp1, function(x) all(is.nan(x)))
ProjectedLeadsImp2 <- ProjectedLeadsImp1[,!bad]
ProjectedLeadsImp3 <-data.frame(cbind(ProjectedLeadsImp2,lockdown))


#Defining our task and learner

n= nrow(ProjectedLeadsImp3)
ProjectedLeadsImp3_train<- ProjectedLeadsImp3[seq(1,n,by=2),]
ProjectedLeadsImp3_test <- ProjectedLeadsImp3[seq(2,n,by=2),]


ProjectedLeadsTask <- makeRegrTask(data= ProjectedLeadsImp3_train, target = "DMA_Projected_Visits_MA3")

lin <- makeLearner("regr.lm")





#Using a filter menthod for feature selection

filterVals <- generateFilterValuesData(ProjectedLeadsTask, method = "linear.correlation")


#Manually selecting which features to drop

ProjectedLeadsFiltTask <- filterFeatures(ProjectedLeadsTask, fval = filterVals, abs = 44)

ProjectedLeadsFiltTask <- filterFeatures(ProjectedLeadsTask, fval = filterVals, perc = 0.03744)

ProjectedLeadsFiltTask <- filterFeatures(ProjectedLeadsTask, fval = filterVals, threshold = 0.3)


#Creating a filter wrapper

filterWrapper = makeFilterWrapper(learner = lin, fw.method = "linear.correlation")

#Tuning the number of predictors to retain
lmParamSpace <- makeParamSet(makeIntegerParam("fw.abs", lower = 1, upper = 44))


gridSearch <- makeTuneControlGrid()

kFold <- makeResampleDesc("CV", iters=10)
tunedFeats <- tuneParams(filterWrapper, task = ProjectedLeadsTask, resampling = kFold, par.set = lmParamSpace, control = gridSearch)


#Training the model with filtered features

filteredTask <- filterFeatures(ProjectedLeadsTask,fval = filterVals, abs = unlist(tunedFeats$x))

filteredModel <- train(lin, filteredTask)


wrapperModelData <- getLearnerModel(filteredModel)
sink("lm.txt")
print(summary(wrapperModelData))
sink()


##############Random Forest Model################

forest <- makeLearner("regr.randomForest")
kFold <- makeResampleDesc("CV", iters = 10)
forestParamSpace <- makeParamSet(makeIntegerParam("ntree", lower = 50, upper = 50),
                                 makeIntegerParam("mtry", lower = 1, upper = 103),
                                 makeIntegerParam("nodesize", lower = 1, upper = 10),
                                 makeIntegerParam("maxnodes", lower = 5,upper = 30))
randSearch <- makeTuneControlRandom(maxit = 100)
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())
tunedForestPars <- tuneParams(forest, task= ProjectedLeadsTask, resampling = kFold, par.set = forestParamSpace, control = randSearch)

parallelStop()
tunedForestPars


tunedForest <- setHyperPars(forest, par.vals = tunedForestPars$x)
tunedForestModel <- train(tunedForest, ProjectedLeadsTask)
forestModelData <- getLearnerModel(tunedForestModel)

plot(forestModelData)



#Building your first Ridge, Lasso, and elastic net models

ProjectedLeadsImp3 <- ProjectedLeadsImp2_test
ProjectedLeadsImp3<-as.data.table(ProjectedLeadsImp3)
ProjectedLeadsTask <- makeRegrTask(data= ProjectedLeadsImp3, target = "DMA_Projected_Visits_MA3")


ridge <- makeLearner("regr.glmnet", alpha = 0, id = "ridge")


ridgeParamSpace <- makeParamSet(makeNumericParam("s", lower = 0, upper = 100))
randSearch <- makeTuneControlRandom(maxit = 200)
cvForTuning <- makeResampleDesc("RepCV", folds=3, reps=10)

library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores())
tunedRidgePars <- tuneParams(ridge, task = ProjectedLeadsTask, resampling = cvForTuning, par.set = ridgeParamSpace, control = randSearch)
parallelStop()


ridgeTuningData <- generateHyperParsEffectData(tunedRidgePars)
plotHyperParsEffect(ridgeTuningData, x= "s", y= "mse.test.mean", plot.type = "line")+theme_bw()


tunedRidge <- setHyperPars(ridge, par.vals = tunedRidgePars$x)
tunedRidgeModel <- train(tunedRidge,ProjectedLeadsTask)


ridgeModelData <- getLearnerModel(tunedRidgeModel,more.unwrap = TRUE)
ridgeCoefs <- coef(ridgeModelData, s= tunedRidgePars$x$s)


#Training the lasso model
lasso <- makeLearner("regr.glmnet", alpha= 1, id= "lasso")
lassoParamSpace <- makeParamSet(makeNumericParam("s", lower = 0, upper = 200))
parallelStartSocket(cpus = detectCores())
tunedLassoPars <- tuneParams(lasso, task = ProjectedLeadsTask, resampling = cvForTuning,par.set = lassoParamSpace, control = randSearch)
parallelStop()
tunedLassoPars

lassoTuningData <- generateHyperParsEffectData(tunedLassoPars)
plotHyperParsEffect(lassoTuningData, x = "s", y = "mse.test.mean", plot.type = "line")+theme_bw()

tunedLasso <- setHyperPars(lasso, par.vals = tunedLassoPars$x)
tunedLassoModel <- train(tunedLasso, ProjectedLeadsTask)

lassoModelData <- getLearnerModel(tunedLassoModel)
lassoCoefs <- coef(lassoModelData, s= tunedLassoPars$x$s)
lassoCoefs

#Training the elastic net model

elastic <- makeLearner("regr.glmnet", id= "elastic")
elasticParamSpace <- makeParamSet(makeNumericParam("s", lower = 0, upper = 200), makeNumericParam("alpha", lower = 0, upper = 1))
randSearchElastic <- makeTuneControlRandom(maxit = 400)
parallelStartSocket(cpus = detectCores())
tunedElasticPars <- tuneParams(elastic, task = ProjectedLeadsTask, resampling = cvForTuning, par.set = elasticParamSpace, control = randSearchElastic)
parallelStop()
tunedElasticPars

tunedElastic <- setHyperPars(elastic, par.vals = tunedElasticPars$x)
tunedElasticModel <- train(tunedElastic, ProjectedLeadsTask)
elasticModelData <- getLearnerModel(tunedElasticModel,more.unwrap = TRUE)
elasticCoefs <- coef(elasticModelData, s= tunedElasticPars$x$s)

sink("lm3.txt")
print(summary(elasticCoefs))
sink()





##XGboost Model

xgb <- makeLearner("regr.xgboost")

xgbParamSpace <- makeParamSet(makeNumericParam("eta", lower = 0, upper = 1),
                              makeNumericParam("gamma", lower = 0, upper = 10),
                              makeIntegerParam("max_depth", lower = 1, upper = 20),
                              makeNumericParam("min_child_weight", lower = 1, upper = 10),
                              makeNumericParam("subsample", lower = 0.5, upper = 1),
                              makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
                              makeIntegerParam("nrounds", lower = 30, upper = 30))

tunedXgbPars <- tuneParams(xgb, task = ProjectedLeadsTask, resampling = kFold, par.set = xgbParamSpace, control = randSearch)
tunedXgbPars



tunedXgb <- setHyperPars(xgb, par.vals = tunedXgbPars$x)
tunedXgbModel <- train(tunedXgb, ProjectedLeadsTask)
xgbModelData <- getLearnerModel(tunedXgbModel)
ggplot(xgbModelData$evaluation_log, aes(iter, train_rmse)) +
  geom_line() +
  geom_point() +
  theme_bw()
tunedXgbModel1<- data.matrix(tunedXgbModel)

outer <- makeResampleDesc("CV", iters = 6)
cvForTuning <- makeResampleDesc("CV", iters = 5)
xgbWrapper <- makeTuneWrapper("regr.xgboost",
                              resampling = cvForTuning,
                              par.set = xgbParamSpace,
                              control = randSearch)

cvWithTuning <- resample(xgbWrapper, ProjectedLeadsTask, resampling = outer)
cvWithTuning

x<-getLearnerModel(cvWithTuning)
sink("lm7.txt")
print(summary(cvWithTuning))
sink()





#####Predictions


####Ridge
test<-as.data.table(ProjectedLeadsImp3_test)
newdata.pred=predict(tunedRidgeModel, newdata= test)
newdata.pred



####Linear Regression



newdata.pred=predict(filteredModel, newdata= ProjectedLeadsImp3_test)
newdata.pred

write.csv(newdata.pred,"linear1.csv")

#####XG Boost



newdata.pred=predict(tunedXgbModel1, ProjectedLeadsImp3_test)
newdata.pred


#####Random Forest

newdata.pred=predict(tunedForestModel, newdata=ProjectedLeadsImp3_test)
newdata.pred
write.csv(newdata.pred,"random1.csv")


#######Elastic Nets

newdata.pred=predict(tunedElasticModel, newdata=ProjectedLeadsImp3_test)
newdata.pred

write.csv(newdata.pred,"elastic.csv")




