library(readr)
library(plyr)
library(magrittr)
library(randomForest)

local_path <- "D:/Personal/Data Science/Titanic Kaggle"
setwd(local_path)

titanic.train <- read.csv("train.csv", stringsAsFactors = FALSE)
titanic.test <- read.csv("test.csv", stringsAsFactors = FALSE)

titanic.full <- rbind.fill(titanic.train, titanic.test)
titanic.full$IsTrainSet <- ifelse(is.na(titanic.full$Survived), FALSE, TRUE)

titanic.full[titanic.full$Embarked == "",]$Embarked <- "S"

age.median <- median(titanic.full$Age, na.rm = TRUE)
titanic.full[is.na(titanic.full$Age),]$Age <- age.median

fare.median <- median(titanic.full$Fare, na.rm = TRUE)
titanic.full[is.na(titanic.full$Fare),]$Fare <- fare.median

factor.cols <- c("Pclass", "Sex", "Embarked")
titanic.full[factor.cols] <- lapply(titanic.full[factor.cols], factor)
str(titanic.full)

titanic.train <- titanic.full[titanic.full$IsTrainSet == TRUE,]
titanic.test <- titanic.full[titanic.full$IsTrainSet == FALSE,]

titanic.train$Survived <- as.factor(titanic.train$Survived)

survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
survived.formula <- as.formula(survived.equation)

titanic.model <- randomForest(formula = survived.formula, data = titanic.train, ntree = 500, mtry = 3, nodesize = 0.01 * nrow(titanic.test))

features.equation <- "Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
titanic.prediction <- predict(titanic.model, newdata = titanic.test)

output.df <- as.data.frame(titanic.test$PassengerId) %>% 
  set_colnames("PassengerId") %>% 
  mutate(Survived = titanic.prediction)

write.csv(output.df, file = "titanic_kaggle_submission.csv", row.names = FALSE)








