# Load necessary libraries
library(e1071)
library(ggplot2)
library(dplyr)
library(caret)

# Load dataset
file_path <- "C:/Users/Sanjay/Desktop/ML Tutorial/archive/laptop_prices.csv"
df <- read.csv(file_path)

# Print column names
print(colnames(df))

# Detect the "Price" column dynamically
price_col <- grep("price", tolower(gsub("[^a-zA-Z0-9]", "", colnames(df))), value = TRUE)

# Ensure the Price column exists
if (length(price_col) == 0) {
  stop("❌ Error: 'Price' column not found in dataset! Check column names above.")
} else {
  cat("✅ Found Price column:", price_col, "\n")
}

# Convert Storage to numeric
df$Storage <- as.numeric(gsub("[^0-9]", "", df$Storage))

# Convert Resolution to total pixel count
df <- df %>%
  separate(Resolution, into = c("Width", "Height"), sep = "x", convert = TRUE, fill = "right") %>%
  mutate(Total_Pixels = as.numeric(Width) * as.numeric(Height)) %>%
  select(-Width, -Height)

# Identify categorical columns
categorical_cols <- c("Brand", "Processor", "GPU", "Operating.System")
categorical_cols <- categorical_cols[categorical_cols %in% colnames(df)]

# Convert categorical columns to factors
if (length(categorical_cols) > 0) {
  df[categorical_cols] <- lapply(df[categorical_cols], as.factor)
}

# Handle missing values
df <- df %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Standardize numeric features
numeric_cols <- df %>% select(where(is.numeric)) %>% colnames()
df[numeric_cols] <- scale(df[numeric_cols])

# Create a binary classification target variable (High Price vs Low Price)
median_price <- median(df[[price_col]], na.rm = TRUE)
df$Price_Category <- ifelse(df[[price_col]] >= median_price, "High", "Low")
df$Price_Category <- as.factor(df$Price_Category)

# Split data into training (80%) and testing (20%)
set.seed(123)
train_index <- createDataPartition(df$Price_Category, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Train an SVM model with radial kernel
svm_model <- svm(Price_Category ~ ., data = train_data, kernel = "radial", cost = 1, gamma = 0.1)

# Make predictions
predictions <- predict(svm_model, test_data)

# Evaluate accuracy
conf_matrix <- confusionMatrix(predictions, test_data$Price_Category)
print(conf_matrix)

# Plot decision boundary using PCA
pca_result <- prcomp(df[, numeric_cols], center = TRUE, scale. = TRUE)
df_pca <- as.data.frame(pca_result$x[, 1:2])
df_pca$Price_Category <- df$Price_Category

ggplot(df_pca, aes(x = PC1, y = PC2, color = Price_Category)) +
  geom_point(size = 3) +
  labs(title = "SVM Classification of Laptop Prices",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal()
