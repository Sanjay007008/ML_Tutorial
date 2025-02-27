# Load necessary libraries
if (!require(randomForest)) install.packages("randomForest", dependencies=TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies=TRUE)
if (!require(caret)) install.packages("caret", dependencies=TRUE)
if (!require(dplyr)) install.packages("dplyr", dependencies=TRUE)

library(randomForest)
library(ggplot2)
library(caret)
library(dplyr)

# Load dataset
file_path <- "C:/Users/Sanjay/Desktop/ML Tutorial/archive/laptop_prices.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# Convert Storage to numerical format (handling missing values)
df$Storage <- as.numeric(gsub("\\D", "", df$Storage))

# Convert Resolution to total pixel count
resolution_split <- strsplit(df$Resolution, "x")
df$Width <- as.numeric(sapply(resolution_split, `[`, 1))
df$Height <- as.numeric(sapply(resolution_split, `[`, 2))
df$Total_Pixels <- df$Width * df$Height
df <- df %>% select(-Resolution, -Width, -Height)

# Encode categorical variables
categorical_cols <- c("Brand", "Processor", "GPU", "Operating.System")
df[categorical_cols] <- lapply(df[categorical_cols], as.factor)

# Define features and target variable
target <- "Price...."
X <- df %>% select(-all_of(target))
y <- df[[target]]

# Split dataset into training and testing sets
set.seed(42)
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

# Train Random Forest Regressor
rf_model <- randomForest(Price.... ~ ., data = train_data, ntree = 100, importance = TRUE, seed = 42)

# Predictions
rf_pred <- predict(rf_model, test_data)

# Evaluate the model
mae <- mean(abs(test_data$Price.... - rf_pred))
mse <- mean((test_data$Price.... - rf_pred)^2)
rmse <- sqrt(mse)
r2 <- cor(test_data$Price...., rf_pred)^2

# Compute Adjusted R²
n <- nrow(test_data)  # Number of test samples
p <- ncol(test_data) - 1  # Number of features
adjusted_r2 <- 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Print accuracy metrics
data.frame(
  Metric = c("Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "R² Score", "Adjusted R² Score", "Final Accuracy (R² Score)"),
  Value = c(round(mae, 2), round(mse, 2), round(rmse, 2), round(r2, 4), round(adjusted_r2, 4), round(r2 * 100, 2))
)

# Feature Importance Visualization
importance_df <- as.data.frame(importance(rf_model))
importance_df$Feature <- rownames(importance_df)
importance_df <- importance_df %>% arrange(desc(IncNodePurity))

ggplot(importance_df, aes(x = reorder(Feature, IncNodePurity), y = IncNodePurity)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Feature Importance in Random Forest", x = "Feature", y = "Importance") +
  theme_minimal()
