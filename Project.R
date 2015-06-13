library(data.table)
library(caret)

# Download and load files into data.tables

setInternet2(TRUE)
setwd("C:/Data/My Project/MOCC/03 Quiz and Projects/08 Machine Learning")

target <- "pml_training.csv"
if (!file.exists(target)) {
    url <-
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    target <- "pml_training.csv"
    download.file(url, destfile = target)
}
training <- read.csv(target, na.strings = c("NA","#DIV/0!",""))

target <- "pml_testing.csv"
if (!file.exists(target)) {
    url <-
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(url, destfile = target)
}
testing <- read.csv(target, na.strings = c("NA","#DIV/0!",""))

# Reduce the number of predictors by removing near zero values, NA, or is empty

# Remove columns with Near Zero Values
subTrain <-
    training[, names(training)[!(nzv(training, saveMetrics = T)[, 4])]]

# Remove columns with NA or is empty
subTrain <-
    subTrain[, names(subTrain)[sapply(subTrain, function (x)
        ! (any(is.na(x) | x == "")))]]


# Remove V1 which seems to be a serial number, and
# cvtd_timestamp that is unlikely to influence the prediction
subTrain <- subTrain[,-1]
subTrain <- subTrain[, c(1:3, 5:58)]

# Divide the training data into a training set and a validation set
inTrain <- createDataPartition(subTrain$classe, p = 0.6, list = FALSE)
subTraining <- subTrain[inTrain,]
subTesting <- subTrain[-inTrain,]

# Perform Principal Component Analysis
# prComp <- prcomp(subTraining[, 4:56])
# subTrainPC <- predict(prComp, subTraining[, 1:56])

# Run Parallel.
# Set up the parallel clusters.

model <- "modelFit.RData"
if (!file.exists(model)) {
    require(parallel)
    require(doParallel)
    cl <- makeCluster(detectCores() - 1)
    registerDoParallel(cl)
    
    # Set the control parameters.
    
    ctrl <- trainControl(
        classProbs = TRUE,
        savePredictions = TRUE,
        allowParallel = TRUE
    )
    
    
    fit <- train(subTraining$classe ~ ., method = "rf", data = subTraining)
    save(fit, file = "modelFit.RData")
    
    stopCluster(cl)
}
elseif {
    load(file = "modelFit.RData", verbose = TRUE)
}

predTrain <- predict(fit, subTraining)
confusionMatrix(predTrain, subTraining$classe)

predTest <- predict(fit, subTesting)
confusionMatrix(predTest, subTesting$classe)

varImp(fit)
fit$finalModel

# The estimated error rate is less than 1%

predTesting <- predict(fit, testing)

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(predTesting)
