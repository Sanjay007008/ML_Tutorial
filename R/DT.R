# Load required libraries
library(rpart)
library(rpart.plot)
library(caret)

# Load dataset
file_path <- "C:/Users/Sanjay/Desktop/ML Tutorial/archive/laptop_prices.csv"  # Updated file path
data <- read.csv(file_path)

# View dataset structure
str(data)
print(colnames(data))  # Check column names to avoid errors

# Handle column name issues
colnames(data) <- gsub("[^[:alnum:]_]", "_", colnames(data))  # Fix special characters in column names

# Convert categorical columns to factors
data$Brand <- as.factor(data$Brand)
data$Processor <- as.factor(data$Processor)
data$GPU <- as.factor(data$GPU)
data$Operating_System <- as.factor(data$Operating_System)

# Convert Storage to numeric
data$Storage <- as.numeric(gsub("[^0-9]", "", data$Storage))

# Convert Resolution to total pixel count
resolution_split <- strsplit(as.character(data$Resolution), "x")
data$Total_Pixels <- sapply(resolution_split, function(x) as.numeric(x[1]) * as.numeric(x[2]))

# Remove unnecessary columns
data <- subset(data, select = -c(Resolution))

# Remove missing values
data <- na.omit(data)

# Ensure target variable is numeric
data$Price <- as.numeric(data$Price)

# Split dataset into training and testing (80-20 split)
set.seed(42)
split <- createDataPartition(data$Price, p = 0.8, list = FALSE)
train_data <- data[split, ]
test_data <- data[-split, ]

# Train Decision Tree Model
decision_tree_model <- rpart(Price ~ ., data = train_data, method = "anova")

# Visualize Decision Tree
rpart.plot(decision_tree_model, main = "Decision Tree for Laptop Prices", type = 3, extra = 101)

# Make Predictions
predictions <- predict(decision_tree_model, test_data)

# Evaluate Model Performance
mse <- mean((test_data$Price - predictions)^2)
rmse <- sqrt(mse)
r2 <- 1 - (sum((test_data$Price - predictions)^2) / sum((test_data$Price - mean(test_data$Price))^2))

# Calculate Accuracy (Custom Approach)
tolerance <- 0.1 * test_data$Price  # 10% tolerance
correct_predictions <- abs(test_data$Price - predictions) <= tolerance
accuracy <- mean(correct_predictions) * 100  # Convert to percentage

# Print Final Accuracy Metrics
cat("\n===== Final Model Evaluation Metrics =====\n")
cat(sprintf("📌 Mean Squared Error (MSE): %.2f\n", mse))
cat(sprintf("📌 Root Mean Squared Error (RMSE): %.2f\n", rmse))
cat(sprintf("📌 R-squared (R²): %.4f\n", r2))
cat(sprintf("✅ Model Accuracy: %.2f%%\n", accuracy))
