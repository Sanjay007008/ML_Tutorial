# Load required libraries
library(e1071)
library(caret)
library(dplyr)

# Load dataset
file_path <- "C:/Users/Sanjay/Desktop/ML Tutorial/archive/laptop_prices.csv"
df <- read.csv(file_path, stringsAsFactors = TRUE)

# Check actual column names
print(colnames(df))  # Check column names before processing

# Fix column names (replace spaces and special characters)
colnames(df) <- gsub("[^[:alnum:]_]", "_", colnames(df))

# Identify the actual column name for price
price_col <- grep("Price", colnames(df), value = TRUE)  # Find column with "Price"

if (length(price_col) == 0) {
  stop("❌ Error: No column related to 'Price' found in the dataset.")
} else {
  cat("✅ Found Price column:", price_col, "\n")
}

# Convert categorical variables to factors
df$Brand <- as.factor(df$Brand)
df$Processor <- as.factor(df$Processor)
df$GPU <- as.factor(df$GPU)
df$Operating_System <- as.factor(df$Operating_System)

# Convert Storage to numeric (extract digits)
df$Storage <- as.numeric(gsub("[^0-9]", "", df$Storage))

# Convert Resolution to total pixel count
resolution_split <- strsplit(as.character(df$Resolution), "x")
df$Total_Pixels <- sapply(resolution_split, function(x) as.numeric(x[1]) * as.numeric(x[2]))

# Drop unnecessary columns
df <- df %>% select(-Resolution)

# Handle missing values
df <- na.omit(df)

# Convert Price to categorical variable (Low, Medium, High)
df$Price_Category <- cut(df[[price_col]],  # Use dynamic column name
                         breaks = quantile(df[[price_col]], probs = c(0, 1/3, 2/3, 1), na.rm = TRUE), 
                         labels = c("Low", "Medium", "High"), 
                         include.lowest = TRUE)

# Remove original Price column
df <- df %>% select(-all_of(price_col))

# Split dataset into training and testing
set.seed(42)
split_index <- createDataPartition(df$Price_Category, p = 0.8, list = FALSE)
train_data <- df[split_index, ]
test_data <- df[-split_index, ]

# Train Naïve Bayes Classifier
nb_model <- naiveBayes(Price_Category ~ ., data = train_data)

# Predictions
predictions <- predict(nb_model, test_data)

# Model Evaluation
accuracy <- mean(predictions == test_data$Price_Category)
cat(sprintf("✅ Model Accuracy: %.2f%%\n", accuracy * 100))

# Confusion Matrix
cat("\n📊 Confusion Matrix:\n")
print(confusionMatrix(predictions, test_data$Price_Category))

