library(tidyverse)
expr <- log2(filtered_gene_expr_top5000_var_t_sorted+1) %>% scale() #Note: plus 1 here
images <- log2(image_feature_t_sorted+0.01) %>% scale()

res_Sparse_CCA <- PMA::CCA(expr, images, typex = "standard", typez = "standard",K=2, standardize=F)

expr_scores <- expr %*% res_Sparse_CCA$u
expr_scores <- as.data.frame(expr_scores)
image_scores <- images %*% res_Sparse_CCA$v
image_scores <- as.data.frame(image_scores)

expr_loadings <- res_Sparse_CCA$u
expr_loadings <- as.data.frame(expr_loadings)