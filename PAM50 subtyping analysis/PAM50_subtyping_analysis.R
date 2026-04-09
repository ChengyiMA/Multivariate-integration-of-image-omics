# Load the PAM50 classifier and robust parameters
library(genefu)
library(dplyr)
library(pheatmap)
data(pam50)
data(pam50.robust)

exp <- read.table("/data/filtered_gene_expr_total_var_t_sorted_tibble.txt", sep='\t', header = TRUE)
exp <- data.frame(exp, row.names = 1)

library(biomaRt)

ensembl_ids <- sub("\\..*$", "", colnames(exp))
mart <- useMart("ensembl", dataset = "hsapiens_gene_ensembl")  # 人类

gene_map <- getBM(
  attributes = c("ensembl_gene_id", "hgnc_symbol"),
  filters = "ensembl_gene_id",
  values = ensembl_ids,
  mart = mart
)

id2symbol <- setNames(gene_map$hgnc_symbol, gene_map$ensembl_gene_id)

new_colnames <- id2symbol[ensembl_ids]
new_colnames[is.na(new_colnames)] <- ensembl_ids[is.na(new_colnames)]

colnames(exp) <- new_colnames
head(exp[1:5,1:5])
dim(exp)

gene_info <- as.data.frame(colnames(exp))
pam50_predictions <- molecular.subtyping(
  sbt.model = "pam50",
  data = exp,
  annot = gene_info,
  do.mapping = FALSE)

# Display the PAM50 subtypes
as.data.frame(pam50_predictions$subtype)

# Display the subtype probabilities
as.data.frame(pam50_predictions$subtype.proba)

# Display the subtypes predictions
as.data.frame(pam50_predictions$subtype.crisp)

# Display the crisp subtypes
m=as.data.frame(pam50_predictions$subtype.proba)


df_subtype.crisp <- as.data.frame(pam50_predictions$subtype.crisp)
df_subtype.crisp$Sample <- rownames(df_subtype.crisp)

library(tidyr)
df_long <- df_subtype.crisp %>%
  pivot_longer(cols = c("Basal","Her2","LumA","LumB","Normal"),
               names_to = "Subtype",
               values_to = "Value") %>%
  filter(Value == 1) %>%
  select(Sample, Subtype)

write.table(
  df_long,
  file = "/data/df_molecular_subtype_by_PAM50.txt",
  sep = "\t",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)

