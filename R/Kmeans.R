# Load required libraries
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)
library(NbClust)
library(readr)
library(tidyr)

# Load dataset
file_path <- "C:/Users/Sanjay/Desktop/ML Tutorial/archive/laptop_prices.csv"
df <- read_csv(file_path)

# Print column names for reference
print(colnames(df))

# Detect the "Price" column dynamically
price_col <- grep("price", tolower(gsub("[^a-zA-Z0-9]", "", colnames(df))), value = TRUE)

# Ensure the Price column exists
if (length(price_col) == 0) {
  stop("❌ Error: 'Price' column not found in dataset! Check column names above.")
} else {
  cat("✅ Found Price column:", price_col, "\n")
}

# Drop Price column
df <- df %>% select(-matches(price_col))

# Convert Storage to numeric (extract numbers safely)
if ("Storage" %in% colnames(df)) {
  df <- df %>%
    mutate(Storage = as.numeric(gsub("[^0-9]", "", Storage)))
}

# Convert Resolution to pixel count (handle missing values)
if ("Resolution" %in% colnames(df)) {
  df <- df %>%
    separate(Resolution, into = c("Width", "Height"), sep = "x", convert = TRUE, fill = "right") %>%
    mutate(Total_Pixels = as.numeric(Width) * as.numeric(Height)) %>%
    select(-Width, -Height)
}

# Identify categorical columns dynamically
categorical_cols <- c("Brand", "Processor", "GPU", "Operating System")
categorical_cols <- categorical_cols[categorical_cols %in% colnames(df)]

# Convert categorical columns to factors
if (length(categorical_cols) > 0) {
  df[categorical_cols] <- lapply(df[categorical_cols], as.factor)
}

# Handle missing values
df <- df %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Convert NaN and Inf values to NA, then replace with median
df <- df %>%
  mutate(across(where(is.numeric), ~ ifelse(is.nan(.), NA, .))) %>%
  mutate(across(where(is.numeric), ~ ifelse(is.infinite(.), NA, .)))

# Final missing value check
df <- df %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# Ensure no NA/NaN/Inf remains
if (any(is.na(df))) stop("❌ Error: NA values still exist in the dataset after preprocessing!")
if (any(is.nan(as.matrix(df %>% select(where(is.numeric)))))) stop("❌ Error: NaN values detected!")
if (any(!is.finite(as.matrix(df %>% select(where(is.numeric)))))) stop("❌ Error: Inf values detected!")

# Remove categorical columns before clustering
df <- df %>% select(where(is.numeric))

# Standardize numeric features
df <- scale(df)

# Find optimal K using the Elbow Method
wss <- sapply(1:10, function(k) {
  kmeans(df, centers = k, nstart = 10)$tot.withinss
})

# Plot Elbow Method graph
plot(1:10, wss, type = "b", pch = 19, col = "blue", 
     xlab = "Number of Clusters (K)", ylab = "Total Within Sum of Squares",
     main = "Elbow Method for Optimal K")

# Choose optimal K
optimal_k <- 3

# Apply K-Means clustering
set.seed(42)
kmeans_model <- kmeans(df, centers = optimal_k, nstart = 10)

# Add cluster labels
df_clustered <- as.data.frame(df)
df_clustered$Cluster <- as.factor(kmeans_model$cluster)

# PCA for visualization
pca_result <- prcomp(df, center = TRUE, scale. = TRUE)
df_pca <- as.data.frame(pca_result$x[, 1:2])
df_pca$Cluster <- df_clustered$Cluster

# Scatter plot of clusters
ggplot(df_pca, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(size = 3) +
  labs(title = paste("K-Means Clustering (K =", optimal_k, ") Visualization"),
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal()

# Compute Silhouette Score
silhouette_score <- silhouette(kmeans_model$cluster, dist(df))

# Plot Silhouette Score
fviz_silhouette(silhouette_score)

# Average Silhouette Score
avg_silhouette <- mean(silhouette_score[, 3])
cat("✅ Average Silhouette Score:", avg_silhouette, "\n")
