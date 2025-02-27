library(caret)
library(randomForest)
library(xgboost)
library(e1071)

file_path <- "C:/Users/Sanjay/Desktop/ML Tutorial/archive/laptop_prices.csv"
df <- read.csv(file_path)

df$Storage <- as.numeric(gsub("\\D", "", df$Storage))

resolution_split <- strsplit(df$Resolution, "x")
df$Width <- as.numeric(sapply(resolution_split, `[`, 1))
df$Height <- as.numeric(sapply(resolution_split, `[`, 2))
df$Total_Pixels <- df$Width * df$Height
df <- df[ , !(names(df) %in% c("Resolution", "Width", "Height"))]

df <- na.omit(df)

df$Brand <- as.numeric(factor(df$Brand))
df$Processor <- as.numeric(factor(df$Processor))
df$GPU <- as.numeric(factor(df$GPU))
df$`Operating.System` <- as.numeric(factor(df$`Operating.System`))

X <- df[, !names(df) %in% c("Price..")]
y <- df$Price..

set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
y_train <- y[train_index]
X_test <- X[-train_index, ]
y_test <- y[-train_index]

decision_tree <- train(y_train ~ ., data = cbind(X_train, y_train), method = "rpart", tuneLength = 10)
random_forest <- randomForest(x = X_train, y = y_train, ntree = 10)

dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

param <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  nrounds = 100,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(params = param, data = dtrain, nrounds = param$nrounds)

meta_train_data <- data.frame(
  DT_pred = predict(decision_tree, X_train),
  RF_pred = predict(random_forest, X_train),
  XGB_pred = predict(xgb_model, newdata = as.matrix(X_train)),
  y_train = y_train
)

meta_model <- lm(y_train ~ ., data = meta_train_data)

predict_dt <- predict(decision_tree, X_test)
predict_rf <- predict(random_forest, X_test)
predict_xgb <- predict(xgb_model, newdata = as.matrix(X_test))

meta_test_data <- data.frame(
  DT_pred = predict_dt,
  RF_pred = predict_rf,
  XGB_pred = predict_xgb
)

meta_pred <- predict(meta_model, newdata = meta_test_data)

mse <- mean((y_test - meta_pred)^2)
r2 <- 1 - sum((y_test - meta_pred)^2) / sum((y_test - mean(y_test))^2)

cat("MSE:", mse, "\nR² Score:", r2, "\n")
cat("Final Accuracy (R² Score):", r2 * 100, "%\n")
