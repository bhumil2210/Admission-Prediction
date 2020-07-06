library(readr)
library(ggplot2)
library(ggthemes)
library(DT)
library(magrittr)
library(dplyr)

library(readr)
library(data.table)
library(plotly)
library(MASS)
library(caret)
library(plyr)
library(e1071)
library(randomForest)

df <- read_csv("Downloads/Admission_Predict.csv")
View(df)
df
head(df)

df = df[complete.cases(df),]
df = na.omit(df)

df$`Serial No.` <- NULL
names(df)[1] <- 'GREScore'
names(df)[2] <- 'TOEFLScore'
names(df)[3] <- 'UNRating'
names(df)[4] <- 'SOP'
names(df)[5] <- 'LOR'
names(df)[6] <- 'CGPA'
names(df)[7] <- 'Research'
names(df)[8] <- 'AdmitProbability'

head(df)

copy_df = df

# Data Analysis and Visualization
str(df)

sum(is.na(df))
library(dplyr)
library(Lahman)
df<-df%>%select(GREScore,TOEFLScore,UNRating,SOP,LOR,CGPA,Research,AdmitProbability)

# Scatter plot, Frequency curve and Correlation values in one graph
library(GGally)
ggpairs(df)

# Histograms
hist(df$GREScore)

# Data Visualization of GRE.Score w.r.t Chance.of.Admit
ggplot(df,aes(x=GREScore,y=AdmitProbability))+geom_point()+geom_smooth()+ggtitle("Chances of Admit vs GRE Score")

# chances of admit vs GRE Score based on research
ggplot(df,aes(x=GREScore,y=AdmitProbability,col=Research))+geom_point()+ggtitle("Chances of Admit vs Gre Score based on Research")

# chances of admit vs GRE Score based on UN Rating
ggplot(df,aes(x=GREScore,y=AdmitProbability,col=UNRating))+geom_point()

# Check if there is any outlier for SOP
hist(df$SOP)

hist(df$LOR)

ggplot(data=df,aes(x=factor(SOP),y=AdmitProbability))+geom_boxplot()

# Research vs admitprobability
ggplot(data=copy_df,aes(x=factor(Research),y=AdmitProbability))+geom_boxplot()

# Correlation Matrix
library(corrplot)

C<-cor(df)
corrplot(C,method='number')

X= df[1:7]
y= df[,8]

library(ISLR)
smp_siz = floor(0.75*nrow(df))
smp_siz

set.seed(123)
train_ind = sample(seq_len(nrow(df)),size = smp_siz)
train = df[train_ind,] 
test = df[-train_ind,]

train_x = train[1:7]
train_y = train[8:8]

test_x = test[1:7]
test_y = test[8:8]


#data.frame(table(df$IndustryVertical))

#linear Regression
fit = lm(AdmitProbability~.,data=train)
predicted = predict(fit,test_x)
format(round(test_y$AdmitProbability, 1), nsmall = 1)
accuracy <- 100*sum(format(round(test_y$AdmitProbability, 1), nsmall = 1) == format(round(predicted, 1), nsmall = 1))/100
print(accuracy)


# Leave one out cross validation - LOOCV
library(caret)
train.control <- trainControl(method = "LOOCV")
model <- train( AdmitProbability~., data = copy_df, method = "lm",
               trControl = train.control)
predicted = predict(model,test_x)
accuracy <- 100*sum(format(round(test_y$AdmitProbability, 1), nsmall = 1) == format(round(predicted, 1), nsmall = 1))/100
print(accuracy)


# K-fold cross-validation

set.seed(123) 
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model <- train(AdmitProbability ~., data = copy_df, method = "lm",
               trControl = train.control)
print(model)
predicted = predict(model,test_x)
accuracy <- 100*sum(format(round(test_y$AdmitProbability, 1), nsmall = 1) == format(round(predicted, 1), nsmall = 1))/100
print(accuracy)

# Repeated K-fold cross-validation

set.seed(123)
train.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 10)
# Train the model
model <- train(AdmitProbability ~., data = copy_df, method = "lm",
               trControl = train.control)
print(model)
predicted = predict(model,test_x)
accuracy <- 100*sum(format(round(test_y$AdmitProbability, 1), nsmall = 1) == format(round(predicted, 1), nsmall = 1))/100
print(accuracy)


# RANDOM FOREST

library(randomForest)
# Fitting model
fit <- randomForest(AdmitProbability ~ .,data=copy_df,ntree=1000)
predicted = predict(fit,test_x)
accuracy <- 100*sum(format(round(test_y$AdmitProbability, 1), nsmall = 1) == format(round(predicted, 1), nsmall = 1))/100
print(accuracy)

# Kmeans Clustering (Partitioning Algorithm)

findingK <- function(data, nc=15, seed=123){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of groups",
       ylab="Sum of squares within a group")}

findingK(copy_df, n = 20)

clusters <- kmeans(copy_df,18)
library(factoextra)
library(cluster)

sil <- silhouette(clusters$cluster, dist(copy_df))
print(mean(sil))
sum <- summary(sil)
fviz_silhouette(sil)

# Hierarchical Algorithm

library(factoextra)
d <- dist(copy_df, method = "euclidean")
fit <- hclust(d, method="ward.D")
plot(fit)
groups <- cutree(fit, k=15)
sil <- silhouette(groups, d)
print(mean(sil))
sum <- summary(sil)
fviz_silhouette(sil)

library(fpc)
cluster.stats(d, clusters$cluster, groups,silhouette = TRUE,compareonly = TRUE) 

cluster.stats(d,clusters$cluster)

library(pvclust)
fit <- pvclust(copy_df, method.hclust="ward.D",
               method.dist="euclidean")
plot(fit)
pvrect(fit, alpha=.95) 

# model based
library(mclust)
fit <- Mclust(copy_df)
plot(fit)

mod4 <- Mclust(copy_df, initialization = list(hcPairs = hc(X, use = "SVD")))
summary(mod4)

# KNN
kNN
fit <-knn3(AdmitProbability~.,train,k=15)
summary(fit)

predicted = predict(fit,test)
predicted
accuracy <- 100*sum(format(round(test_y$AdmitProbability, 1), nsmall = 1) == format(round(predicted, 1), nsmall = 1))/100
print(accuracy)
