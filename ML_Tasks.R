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

googleTask <- makeRegrTask(data= googleImp2, target = "DMA_Google_Queries_Index_Adj")

lin <- makeLearner("regr.lm")

#Using a filter menthod for feature selection

filterVals <- generateFilterValuesData(googleTask, method = "linear.correlation")


#Manually selecting which features to drop

googleFiltTask <- filterFeatures(googleTask, fval = filterVals, abs = 44)

googleFiltTask <- filterFeatures(googleTask, fval = filterVals, perc = 0.03744)

googleFiltTask <- filterFeatures(googleTask, fval = filterVals, threshold = 0.3)


#Creating a filter wrapper

filterWrapper = makeFilterWrapper(learner = lin, fw.method = "linear.correlation")

#Tuning thr number of predictors to retain
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



wrapperModelData <- getLearnerModel(wrapperModel)
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







#Building Ridge, Lasso, and elastic net models

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


ridgeModelData <- getLearnerModel(tunedRidgeModel)
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
elasticModelData <- getLearnerModel(tunedElasticModel)
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
