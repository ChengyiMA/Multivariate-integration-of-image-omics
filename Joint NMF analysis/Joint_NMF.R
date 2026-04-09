#l2 normalise
l2_normalize_cols <- function(X) {
  norms <- sqrt(colSums(X^2))
  X_normalized <- sweep(X, 2, norms, "/")
  return(X_normalized)
}

expr_col_l2 <- l2_normalize_cols(filtered_gene_expr_top5000_var_t_sorted)
image_col_l2 <- l2_normalize_cols(image_feature_t_sorted)

X1_expr <- as.matrix(expr_col_l2)
X2_image <- as.matrix(image_col_l2)


BiNMF_mm_2matrix <- function(X1, X2, K, maxiter = 10000, speak = TRUE) {
  # Input：
  # X1 (n x m1), X2 (n x m2) -- Non-negative matrices
  # K       -- Number of components
  # maxiter -- Maximum interation number
  # speak   -- Whether print progress
  
  if (any(X1 < 0) || any(X2 < 0)) {
    stop("Input matrix elements cannot be negative")
  }
  
  n1 <- nrow(X1)
  m1 <- ncol(X1)
  n2 <- nrow(X2)
  m2 <- ncol(X2)
  
  if (n1 != n2) {
    stop("Input matrices should have the same number of rows")
  }
  n <- n1
  
  # Initialising W, H1, H2，random non-negative numbers
  # set.seed(123) #，ensure reproducible outcome
  W <- matrix(runif(n * K), nrow = n, ncol = K)
  H1 <- matrix(runif(K * m1), nrow = K, ncol = m1)
  H2 <- matrix(runif(K * m2), nrow = K, ncol = m2)
  
  eps <- .Machine$double.eps
  
  Xr_old1 <- W %*% H1
  Xr_old2 <- W %*% H2
  
  nmf_euclidean_dist <- function(X, Y) {
    sum((X - Y)^2)
  }
  
  for (iter in 1:maxiter) {
    # update H1, H2
    numerator_H1 <- t(W) %*% X1
    denominator_H1 <- (t(W) %*% W) %*% H1 + eps
    H1 <- H1 * (numerator_H1 / denominator_H1)
    
    numerator_H2 <- t(W) %*% X2
    denominator_H2 <- (t(W) %*% W) %*% H2 + eps
    H2 <- H2 * (numerator_H2 / denominator_H2)
    
    # update W
    combined_H <- cbind(H1, H2)  # K x (m1 + m2)
    combined_X <- cbind(X1, X2)  # n x (m1 + m2)
    
    numerator_W <- combined_X %*% t(combined_H)
    denominator_W <- W %*% (combined_H %*% t(combined_H)) + eps
    W <- W * (numerator_W / denominator_W)
    
    # print progress
    if ((iter %% 20 == 0) && speak) {
      Xr1 <- W %*% H1
      Xr2 <- W %*% H2
      diff <- sum(abs(Xr_old1 - Xr1)) + sum(abs(Xr_old2 - Xr2))
      Xr_old1 <- Xr1
      Xr_old2 <- Xr2
      
      eucl_dist1 <- nmf_euclidean_dist(X1, Xr1)
      eucl_dist2 <- nmf_euclidean_dist(X2, Xr2)
      eucl_dist <- eucl_dist1 + eucl_dist2
      
      errorx1 <- mean(abs(X1 - Xr1)) / mean(X1)
      errorx2 <- mean(abs(X2 - Xr2)) / mean(X2)
      errorx <- errorx1 + errorx2
      
      cat(sprintf("Iter = %d, relative error = %.6e, diff = %.6e, eucl dist = %.6e\n",
                  iter, errorx, diff, eucl_dist))
      
      if (errorx < 1e-5) break
    }
  }
  return(list(W = W, H1 = H1, H2 = H2))
}

res_joint_NMF <- BiNMF_mm_2matrix(X1_expr, X2_image, K = 2, maxiter = 1000, speak = FALSE)

library(cluster)   # silhouette
library(ggplot2)
library(factoextra)
library(gridExtra)

k_range <- 2:50
sil_scores <- numeric(length(2:50))

for (i in seq_along(k_range)) {
  k <- k_range[i]
  
  # W: n x k (n: sample number)
  
  nmf_result <- BiNMF_mm_2matrix(X1_expr, X2_image, k, maxiter = 1000, speak = FALSE)  # Defined BiNMF_mm_2matrix
  W <- nmf_result$W
  
  # clustering based on row vectors in W（usually using kmeans）
  km <- kmeans(W, centers = k, nstart = 20)
  sil <- silhouette(km$cluster, dist(W))
  sil_scores[i] <- mean(sil[, 3])
}

df_sil_scores <- data.frame(k = k_range, sil = sil_scores)

ggplot(df_sil_scores, aes(x = k, y = sil)) +
  geom_point(color = "blue", size = 3) +
  geom_line(color = "blue", linewidth = 1) +
  labs(x = "Number of Factors (k)", y = "Average Silhouette Score") +
  theme_minimal()

