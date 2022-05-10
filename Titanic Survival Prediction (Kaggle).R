library(readr)
library(plyr)
library(magrittr)
library(randomForest)
library(caret)

local_path <- "D:/Personal/Data Science/Titanic Kaggle"
setwd(local_path)

titanic.train <- read.csv("train.csv", stringsAsFactors = FALSE)
titanic.test <- read.csv("test.csv", stringsAsFactors = FALSE)

titanic.full <- rbind.fill(titanic.train, titanic.test)
titanic.full$IsTrainSet <- ifelse(is.na(titanic.full$Survived), FALSE, TRUE)

titanic.full[titanic.full$Embarked == "",]$Embarked <- "S"

age.median <- median(titanic.full$Age, na.rm = TRUE)
titanic.full[is.na(titanic.full$Age),]$Age <- age.median

# Fill in missing value in Age column using OLS
upper.whisker <- boxplot.stats(titanic.full$Age)$stats[5]
lower.whisker <- boxplot.stats(titanic.full$Age)$stats[1]
outlier.filter <- titanic.full$Age >= lower.whisker & titanic.full$Age <= upper.whisker

age.equation = "Age ~ Pclass + Sex + Fare + SibSp + Parch + Embarked"
age.model <- lm(
  formula = age.equation,
  data = titanic.full[outlier.filter,]
)
age.row <- titanic.full[
  is.na(titanic.full$Age),
  c("Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked")
]
age.prediction <- predict(age.model, newdata = age.row)
titanic.full[is.na(titanic.full$Age),]$Age <- age.prediction

# Fill in missing value in Fare column using OLS
upper.whisker <- boxplot.stats(titanic.full$Fare)$stats[5]
outlier.filter <- titanic.full$Fare <= upper.whisker

fare.equation = "Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked"
fare.model <- lm(
  formula = fare.equation,
  data = titanic.full[outlier.filter,]
)
fare.row <- titanic.full[
  is.na(titanic.full$Fare),
  c("Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked")
]
fare.prediction <- predict(fare.model, newdata = fare.row)
titanic.full[is.na(titanic.full$Fare),]$Fare <- fare.prediction

# Predict Survival
factor.cols <- c("Pclass", "Sex", "Embarked")
titanic.full[factor.cols] <- lapply(titanic.full[factor.cols], factor)
str(titanic.full)

titanic.train <- titanic.full[titanic.full$IsTrainSet == TRUE,]
titanic.test <- titanic.full[titanic.full$IsTrainSet == FALSE,]

titanic.train$Survived <- as.factor(titanic.train$Survived)

survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
survived.formula <- as.formula(survived.equation)

titanic.model <- randomForest(formula = survived.formula, 
                              data = titanic.train, 
                              ntree = 150, 
                              mtry = 6, 
                              nodesize = 0.01 * nrow(titanic.test),
                              importance = TRUE,
                              proximity = TRUE)

## Tune the model parameters (mtry and ntree) by tuneRF and plot
# plot(titanic.model)
# survived.col <- grep("Survived", colnames(titanic.train))
# t <- tuneRF(titanic.train[, -survived.col], titanic.train[, survived.col],
#        stepFactor = 0.5,
#        plot = TRUE,
#        ntreeTry = 150,
#        trace = TRUE,
#        improve = 0.05)

features.equation <- "Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
titanic.prediction <- predict(titanic.model, newdata = titanic.test)

output.df <- as.data.frame(titanic.test$PassengerId) %>% 
  set_colnames("PassengerId") %>% 
  mutate(Survived = titanic.prediction)

write.csv(output.df, file = "titanic_kaggle_submission.csv", row.names = FALSE)
