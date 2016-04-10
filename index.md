#Practical Machine Learning Course Project#
###By: Aditya###
####Date: 10 April 2016####



This Project has the aim of understanding and predicting the manner in which ***six*** participants did their exercise using ***dumbells***. The data was collected using devices such as ***Jawbone Up, Nike FuelBand, and Fitbit***. The **Human Activity Recognition** has become an important area of research in recent times and and focusses on how well a given activity was performed by the user.



The first step is to download the  data in `CSV` format and read it into R:

```r
## Downloading and reading the Activity Data:
## Training URL:
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
## Testing URL:
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
## Reading the Training Data:
train <- read.csv(url(url1))
## Reading the Testing Data:
test <- read.csv(url(url2))
```

Now that the data has been read into R, the next step is cleaning the data. Upon looking at the data structure, it can be understood that there are many columns with mostly `NA`. The first step is to remove these columns. 

```r
## Finding and removing columns with mostly NA values:
naCol <- sapply(train, function(x) any(is.na(x)|x == " "))
train <- train[, naCol == FALSE]
test <- test[, naCol == FALSE]
```

Now, removing any column with near zero variances, and columns which are not relevant to this analysis:

```r
## Finding and removing columns with NZV values:
zeroVar <- nearZeroVar(train, saveMetrics = TRUE)
train <- train[, zeroVar$nzv == FALSE]
zeroVar1 <- nearZeroVar(test, saveMetrics = TRUE)
test <- test[, zeroVar1$nzv == FALSE]
## Removing the first six columns from the resultanat dataset since they are not relevant for preiction:
train <- train[, 7:length(colnames(train))]
test <- test[, 7:length(colnames(test))]
```

Now that the data is ready for processing, the training dataset will be divided into Training and Test sets (70:30). Then, the **Random Forest Model (RF)** will be used to analyze and predict the activities:

```r
## Dividing the training dataset:
set.seed(1323)
inTrain <- createDataPartition(y = train$classe, p = 0.70, list = FALSE)
Training <- train[inTrain, ]
Testing <- train[-inTrain, ]
## Building the RF model:
set.seed(13232)
rfModel <- randomForest(classe~., data = Training)
## Predicting the Tesing data with RF model,
predicted <- predict(rfModel, Testing, type = "class")
## Checking the accuracy of predictions using Confusion Matrix
confusionMatrix(predicted, Testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669    4    0    0    0
##          B    4 1135    6    0    0
##          C    0    0 1020    8    1
##          D    0    0    0  956    2
##          E    1    0    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9956          
##                  95% CI : (0.9935, 0.9971)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9944          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9965   0.9942   0.9917   0.9972
## Specificity            0.9991   0.9979   0.9981   0.9996   0.9998
## Pos Pred Value         0.9976   0.9913   0.9913   0.9979   0.9991
## Neg Pred Value         0.9988   0.9992   0.9988   0.9984   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2836   0.1929   0.1733   0.1624   0.1833
## Detection Prevalence   0.2843   0.1946   0.1749   0.1628   0.1835
## Balanced Accuracy      0.9980   0.9972   0.9961   0.9956   0.9985
```

```r
## Below is a plot of the Random Forest Model
plot(rfModel)
```

![](index_files/figure-html/unnamed-chunk-5-1.png)

The above model predicted the Training data with a very high accuracy of ***99.56%***. However, to test if accuracy can be further improved, the **Decision Tree model** can be used:

```r
## Building the tree model
set.seed(13232)
treeModel <- rpart(classe~., data = Training, method = "class")
## Predicting with the model
predicted1 <- predict(treeModel, Testing, type = "class")
## Checking the accuracy of predictions using Confusion Matrix
confusionMatrix(predicted1, Testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1533  198   75   99   22
##          B   55  688   58   82   71
##          C   32  143  812  101  146
##          D   40   81   80  566   77
##          E   14   29    1  116  766
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7417          
##                  95% CI : (0.7303, 0.7529)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6716          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9158   0.6040   0.7914  0.58714   0.7079
## Specificity            0.9064   0.9440   0.9132  0.94351   0.9667
## Pos Pred Value         0.7955   0.7212   0.6580  0.67062   0.8272
## Neg Pred Value         0.9644   0.9085   0.9540  0.92105   0.9363
## Prevalence             0.2845   0.1935   0.1743  0.16381   0.1839
## Detection Rate         0.2605   0.1169   0.1380  0.09618   0.1302
## Detection Prevalence   0.3274   0.1621   0.2097  0.14342   0.1573
## Balanced Accuracy      0.9111   0.7740   0.8523  0.76532   0.8373
```

The accuracy level of the Decision Tree model above is only ***74.17%*** which is much lower than the Random Forest Model. It is obvious that the Random Forest Model gives superior predictions compared to the Decision Tree model. 
