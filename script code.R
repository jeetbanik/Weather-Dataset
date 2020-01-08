# Dataset links downloadable from Kaggle # 
# https://www.kaggle.com/vonline9/weather-istanbul-data-20092019/data
# https://www.kaggle.com/vonline9/weather-istanbul-data-20092019/download
# https://www.kaggle.com/vonline9/weather-istanbul-data-20092019/download/cr3DbJpST7Y7iCmTUn9R%2Fversions%2FBBPE8tsU60Y5CukZshHu%2Ffiles%2FIstanbul%20Weather%20Data.csv?datasetVersionNumber=2
# Need to login to Kaggle to download the dataset thus didn't download directly in R #

# Load required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(Amelia)) install.packages("Amelia", repos = "http://cran.us.r-project.org")
if(!require(mice)) install.packages("mice", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(klaR)) install.packages("klaR", repos = "http://cran.us.r-project.org")
if(!require(httpuv)) install.packages("httpuv", repos = "http://cran.us.r-project.org")
if(!require(class)) install.packages("class", repos = "http://cran.us.r-project.org")
if(!require(Metrics)) install.packages("Metrics", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

# Get the current working directory to copy the downloaded dataset here to import it
getwd()

# Read the data into a data frame 'df'
df <- read.csv("Istanbul Weather Data.csv")

View(df)

# Data analysis
head(df)
str(df)
summary(df)

# Naive Bayes Model

# Create a copy of the original data frame to work with in this model
dat1 <- df

# Data cleaning
dat1 <- dat1[,c(-1,-6:-9)]
dat1$Condition <- ifelse(dat1$Condition=="Sunny", T, F)
dat1$Condition <- factor(dat1$Condition, levels=c(F, T))

head(dat1)

# Convert '0' values into NA
dat1[,2:4][dat1[,2:4]==0] <- NA

# Visualize the missing data NA
missmap(dat1)

# Use mice function to predict the missing values
m <- mice(dat1[, c("Rain","MinTemp","MaxTemp")], method='rf')
m1 <- complete(m)

# Move the predicted missing values into the main dataset
dat1$Rain <- m1$Rain
dat1$MinTemp <- m1$MinTemp
dat1$MaxTemp <- m1$MaxTemp

# Data Visualization
missmap(dat1)
# Distribution of rain by condition
ggplot(dat1, aes(Rain, colour=Condition)) +
  geom_freqpoly(binwidth=1) + labs(title="Rain Distribution by Condition")
# Distribution of minimum tempearture by condition
ggplot(dat1, aes(x=MinTemp, fill=Condition, color=Condition)) +
  geom_histogram(binwidth=1) + labs(title="MinTemp Distribution by Condition") +
  theme_bw()
# Distribution of maximum temperature by condition
ggplot(dat1, aes(x=MaxTemp, fill=Condition, color=Condition)) +
  geom_histogram(binwidth=1) + labs(title="MaxTemp Distribution by Condition") +
  theme_bw()
# Distribution of average humidity by condition
ggplot(dat1, aes(AvgHumidity, colour=Condition)) +
  geom_freqpoly(binwidth=1) + labs(title="AvgHumidity Distribution by Condition")

# Split the data into train and test datasets
indextrain <- createDataPartition(y=dat1$Condition, p=0.8, list=F)
train1 <- dat1[indextrain,]
test1 <- dat1[-indextrain,]

# Check dimensions of the split
prop.table(table(dat1$Condition)) * 100
prop.table(table(train1$Condition)) * 100
prop.table(table(test1$Condition)) * 100

# Create objects x and y holding the predictor and the response variables respectively
x = train1[,-1]
y = train1$Condition

# Apply Naive Bayes
nb_model <- naiveBayes(Condition ~ ., data=train1)
summary(nb_model)

# Predict test set
predict <- predict(nb_model, newdata=test1[-1])

# Get the confusion matrix to see the accuracy value and other parameter values
new1 <- data.frame("Rain"=44,"MaxTemp"=29,"MinTemp"=23,"AvgWind"=19,"AvgHumidity"=57,"AvgPressure"=1017)
c1 <- predict(nb_model, new1)

if (c1==TRUE) { 
  print('Sunny')
} else {
  print('Rainy')
}

# Calculate the accuracy
a1 <- mean(test1[,1]==predict)
a1

# Add accuracy results in a data frame 'acc'
acc <- data.frame(Method="Naive Bayes Model", Accuracy=a1)
acc

#KNN Model

# Create a copy of the original data frame to work with in this model
dat2 <- df

# Data cleaning
dat2 <- dat2[,-1]
dat2$AvgWind <- as.numeric(dat2$AvgWind)
dat2$MoonRise <- as.numeric(dat2$MoonRise)
dat2$MoonSet <- as.numeric(dat2$MoonSet)
dat2$SunRise <- as.numeric(dat2$SunRise)
dat2$SunSet <- as.numeric(dat2$SunSet)
dat2$Rain <- as.numeric(dat2$Rain)
dat2$MinTemp <- as.numeric(dat2$MinTemp)
dat2$AvgHumidity <- as.numeric(dat2$AvgHumidity)
dat2$MaxTemp <- as.numeric(dat2$MaxTemp)
dat2$AvgPressure<-as.numeric(dat2$AvgPressure)

head(dat2)

r <- sample(1:nrow(dat2), 0.9*nrow(dat2)) 

# Create normalization function
norm <-function(x){
  (x-min(x))/(max(x)-min(x))}

# Run normalization on first 10 columns of the dataset as they are the predictors
dat2_norm <- as.data.frame(lapply(dat2[,c(2,3,4,5,6,7,8,9,10,11)], norm))
summary(dat2_norm)
train2 <- dat2_norm[r,] 

# Extract test set
test2 <- dat2_norm[-r,]

# Extract 5th column of train dataset as it'll be used as 'cl' argument in knn function.
t1 <- dat2[r,1]

# Extract 5th column of test dataset to measure the accuracy
t2 <- dat2[-r,1]

# Run knn function
p <- knn(train2, test2, cl=t1, k=6)

# Calculate the accuracy
a2 <- mean(p==t2)
a2

# Save accuracy results in the data frame 'acc'
acc <- bind_rows(acc, data_frame(Method="KNN Model", Accuracy=a2))
acc

# Random Forest Model

set.seed(100, sample.kind="Rounding")

# Create a copy of the original data frame to work with in this model
dat3 <- df

# Data cleaning
dat3 <- dat3[,-1]
dat3$Condition <- ifelse(dat3$Condition =="Sunny", 1, 0)
dat3$Condition <- factor(dat3$Condition, levels = c(0, 1))
dat3$AvgWind <- as.integer(dat3$AvgWind)
dat3$MoonRise <- as.integer(dat3$MoonRise)
dat3$MoonSet <- as.integer(dat3$MoonSet)
dat3$SunRise <- as.integer(dat3$SunRise)
dat3$SunSet <- as.integer(dat3$SunSet)
dat3$Rain <- as.integer(dat3$Rain)
dat3$MinTemp <- as.integer(dat3$MinTemp)
dat3$AvgHumidity <- as.integer(dat3$AvgHumidity)
dat3$MaxTemp <- as.integer(dat3$MaxTemp)
dat3$AvgPressure <- as.integer(dat3$AvgPressure)

head(dat3)

# Split the dataset into train and test
train3<-dat3[1:3000,]
test3<-dat3[3001:3854,]

# Apply Random Forest
rf_model <- randomForest(Condition ~., data = train3)
rf_model
importance(rf_model)

# Predict test set
pred <- predict(rf_model, newdata=test3[,-1], type ='class')

# Get the confusion matrix to see the accuracy value and other parameter values
new2 <- data.frame("Rain"= 0,"MaxTemp"= 29,"MinTemp"= 23,"AvgWind"= 19,"AvgHumidity"= 57,"AvgPressure"=1017,"SunRise"=20,"SunSet"=175,"MoonRise"=934,"MoonSet"=165)
c2 <- predict(rf_model, new2)

if (c2==1) { 
  print('Sunny')
} else  {
  print('Rainy')
}

# Calculate the accuracy
a3 <- auc(pred,test3$Condition)
a3

# Save accuracy results in the data frame 'acc'
acc <- bind_rows(acc, data_frame(Method="Random Forest Model", Accuracy=a3))
acc

# Check accuracy results
acc %>% knitr::kable()
# Most accurate model is the Random Forest Model with an accuracy score of 0.8198043 #