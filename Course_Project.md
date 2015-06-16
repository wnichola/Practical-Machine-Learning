# Practical Machine Learning: Course Project
Nicholas Wee  
Saturday, June 13, 2015  

## Download and load files into data.tables
Check if the data files exist.  If not, download it from the URL.  When reading in the file, change invalid values to NA.  


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
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
```

## Remove invalid predictors
Reduce the number of predictors by removing columns that have near zero values, NA, or is empty.


```r
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
```
## Separate the data to be used for Cross Validation
Using the training data, separate out a set to be used for validation.  From what I've read, there is no need to create a separate set for validation given the Random Forests algorithm is a classifier based on primarily two methods - bagging and random subspace method.  And the datasets used are created using bootstrapping (resampling with replacement), and internally performs cross-validation to refine the model. The out-of-bag error estimate defines the estimation errors in the internal generated validation sets.   

However, as it is one of the project evaluation criteria, there is no harm to create a cross validation dataset to compare the model created by the training subset.  


```r
# Divide the training data into a training set and a validation set
inTrain <- createDataPartition(subTrain$classe, p = 0.6, list = FALSE)
subTraining <- subTrain[inTrain,]
subValidation <- subTrain[-inTrain,]
```

## Create the prediction model (using random forest)
Using the first set of data, to create the prediction model (using random forest).  

Because of the CPU resources required, first, setup to run it in Parallel, using all the CPU cores available...as advised by Course Forum.  Also, even with it running on multiple cores, it will take a long time to create the prediction model. As such, load the model from a previous good run by checking if the model file exists.  
  

```r
# Check if model file exists
model <- "modelFit.RData"
if (!file.exists(model)) {

    # If not, set up the parallel clusters.  
    require(parallel)
    require(doParallel)
    cl <- makeCluster(detectCores() - 1)
    registerDoParallel(cl)
    
    fit <- train(subTraining$classe ~ ., method = "rf", data = subTraining)
    save(fit, file = "modelFit.RData")
    
    stopCluster(cl)
} else {
    # Good model exists from previous run, load it and use it.  
    load(file = "modelFit.RData", verbose = TRUE)
}
```

```
## Loading objects:
##   fit
```

## Measure the Accuracy and Sample Error of the prediction model
Using the training subset and create a prediction.  Then measure it's accuracy


```r
predTrain <- predict(fit, subTraining)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
confusionMatrix(predTrain, subTraining$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    1    0    0    0
##          B    0 2278    1    0    0
##          C    0    0 2053    2    0
##          D    0    0    0 1927    1
##          E    0    0    0    1 2164
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9995          
##                  95% CI : (0.9989, 0.9998)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9994          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9996   0.9995   0.9984   0.9995
## Specificity            0.9999   0.9999   0.9998   0.9999   0.9999
## Pos Pred Value         0.9997   0.9996   0.9990   0.9995   0.9995
## Neg Pred Value         1.0000   0.9999   0.9999   0.9997   0.9999
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1934   0.1743   0.1636   0.1838
## Detection Prevalence   0.2844   0.1935   0.1745   0.1637   0.1838
## Balanced Accuracy      0.9999   0.9997   0.9997   0.9992   0.9997
```

Using the validation subset and create a prediction.  Then measure it's accuracy.  From the training subset, the accuracy is very high, at above 99%.  The sample error is 0.0008.


```r
predValidation <- predict(fit, subValidation)
confusionMatrix(predValidation, subValidation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    0 1518    4    0    0
##          C    0    0 1364    1    0
##          D    0    0    0 1285    0
##          E    0    0    0    0 1442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9994          
##                  95% CI : (0.9985, 0.9998)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9992          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9971   0.9992   1.0000
## Specificity            1.0000   0.9994   0.9998   1.0000   1.0000
## Pos Pred Value         1.0000   0.9974   0.9993   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   0.9994   0.9998   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1935   0.1738   0.1638   0.1838
## Detection Prevalence   0.2845   0.1940   0.1740   0.1638   0.1838
## Balanced Accuracy      1.0000   0.9997   0.9985   0.9996   1.0000
```

From the validation subset, the accuracy is still very high, at above 99%, with an out-of-sample error of 0.0003.  Not significantly different from the sample error.  
  
From the model, the following are the list of important predictors in the model.  


```r
varImp(fit)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 60)
## 
##                      Overall
## raw_timestamp_part_1 100.000
## num_window            52.431
## roll_belt             48.479
## pitch_forearm         30.725
## yaw_belt              22.734
## magnet_dumbbell_z     22.376
## magnet_dumbbell_y     18.392
## pitch_belt            17.774
## roll_forearm          11.973
## roll_dumbbell          7.898
## accel_dumbbell_y       7.703
## magnet_dumbbell_x      7.169
## accel_belt_z           6.980
## accel_forearm_x        6.846
## total_accel_dumbbell   5.837
## magnet_belt_y          5.807
## accel_dumbbell_z       5.729
## magnet_belt_z          5.127
## accel_dumbbell_x       3.460
## yaw_dumbbell           3.434
```

```r
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 31
## 
##         OOB estimate of  error rate: 0.12%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3348    0    0    0    0 0.0000000000
## B    1 2277    1    0    0 0.0008775779
## C    0    5 2048    1    0 0.0029211295
## D    0    0    3 1926    1 0.0020725389
## E    0    0    0    2 2163 0.0009237875
```

The reported OOB Estimated Error is at 12%.  However, based on the validation accuracy at over 99%, with CI between 99.87% to 99.97%, the prediction model can be applied to the final testing set, and predict the classe in the 20 test cases.  

## Apply the prediction model
Apply the prediction model to the testing data. The predicted classification are:  


```r
predTesting <- predict(fit, testing)
predTesting
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

And generate the files for submission with the given R code from the assignment.


```r
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(predTesting)
```

   
   ## End
