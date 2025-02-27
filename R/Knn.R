# Load required libraries (install only if missing)
packages <- c("tidyverse", "class", "caret", "ggplot2")
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg, dependencies = TRUE)
  library(pkg, character.only = TRUE)
}
lapply(packages, install_if_missing)

# Load dataset
file_path <- "C:/Users/Sanjay/Desktop/ML Tutorial/archive/laptop_prices.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# Print column names to verify structure
print("Column names in dataset:")
print(colnames(df))

# Ensure Operating System column exists
actual_os_col <- colnames(df)[grepl("Operating", colnames(df), ignore.case = TRUE)]
if (length(actual_os_col) == 0) {
  stop("Error: Column related to Operating System not found in dataset. Check column names.")
} else {
  colnames(df)[colnames(df) == actual_os_col] <- "Operating_System"
}

# Ensure Price column exists
actual_price_col <- colnames(df)[grepl("Price", colnames(df), ignore.case = TRUE)]
if (length(actual_price_col) == 0) {
  stop("Error: Column related to Price not found in dataset. Check column names.")
} else {
  colnames(df)[colnames(df) == actual_price_col] <- "Price"
}

# Ensure Storage is numeric (remove non-numeric characters)
df$Storage <- as.numeric(gsub("[^0-9]", "", df$Storage))

# Convert Resolution to total pixel count (if column exists)
if ("Resolution" %in% colnames(df)) {
  resolution_split <- strsplit(as.character(df$Resolution), "x")
  df$Total_Pixels <- sapply(resolution_split, function(x) as.numeric(x[1]) * as.numeric(x[2]))
  df <- df %>% select(-Resolution)  # Remove original Resolution column
}

# Print first few rows to check structure
print(head(df))

# Handle missing values before encoding
df <- df %>% drop_na(Brand, Processor, GPU, Operating_System, Price)

# Encode categorical variables
df$Brand <- as.factor(df$Brand)
df$Processor <- as.factor(df$Processor)
df$GPU <- as.factor(df$GPU)
df$Operating_System <- as.factor(df$Operating_System)

# Convert Price into categories (Low, Medium, High)
df$Price_Category <- cut(df$Price, 
                         breaks = quantile(df$Price, probs = seq(0, 1, by = 1/3), na.rm = TRUE), 
                         labels = c("Low", "Medium", "High"), 
                         include.lowest = TRUE)

# Remove rows with missing values
df <- na.omit(df)

# Define features (X) and target variable (y)
X <- df %>% select(-c(Price, Price_Category))

# 🔹 Fix: Select only numeric columns for scaling
X_numeric <- X %>% select(where(is.numeric))

# Check column types
print("Checking feature data types:")
print(str(X_numeric))

# Normalize features
X_scaled <- as.data.frame(scale(X_numeric))

# Split dataset into training and testing sets (80% train, 20% test)
set.seed(42)
train_indices <- createDataPartition(df$Price_Category, p = 0.8, list = FALSE)
X_train <- X_scaled[train_indices, ]
X_test <- X_scaled[-train_indices, ]
train_labels <- df$Price_Category[train_indices]
test_labels <- df$Price_Category[-train_indices]

# Ensure labels are factors with the same levels
train_labels <- factor(train_labels)
test_labels <- factor(test_labels, levels = levels(train_labels))

# Train KNN model with k=5
k <- 5
knn_pred <- knn(train = X_train, test = X_test, cl = train_labels, k = k)

# Evaluate model performance
accuracy <- mean(knn_pred == test_labels) * 100
cat("KNN Model Accuracy:", round(accuracy, 2), "%\n")

# Confusion matrix
conf_matrix <- confusionMatrix(knn_pred, test_labels)
print(conf_matrix)

# Plot accuracy for different K values
k_values <- seq(1, 20, by = 1)
accuracies <- sapply(k_values, function(k) {
  pred_k <- knn(train = X_train, test = X_test, cl = train_labels, k = k)
  mean(pred_k == test_labels)
})

# Plot the KNN accuracy curve
ggplot(data.frame(K = k_values, Accuracy = accuracies), aes(x = K, y = Accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  ggtitle("K-Value vs Accuracy for KNN") +
  xlab("Number of Neighbors (K)") +
  ylab("Accuracy") +
  theme_minimal()
