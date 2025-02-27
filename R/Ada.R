library(xgboost)
library(caret)

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

predictions <- predict(xgb_model, newdata = dtest)

mse <- mean((y_test - predictions)^2)
r2 <- 1 - sum((y_test - predictions)^2) / sum((y_test - mean(y_test))^2)

cat("MSE:", mse, "\nR² Score:", r2, "\n")
cat("Final Accuracy (R² Score):", r2 * 100, "%\n")
