# 1. Create a vector of column names from logit_diff_llava that match the head-layer combinations in fact_heads
head_layer_combinations <- with(fact_heads, paste("logit_lens_head_out_L", layer, "H", head, sep = ""))

# 2. Subset logit_diff_llava to keep only the relevant columns
logit_subset <- logit_diff_llava[ , colnames(logit_diff_llava) %in% head_layer_combinations]

# 3. Convert the subset to numeric (if not already numeric) and flatten to a vector
logit_subset_numeric <- as.numeric(unlist(logit_subset))

# 4. Compute the average of the selected columns
average_value <- mean(logit_subset_numeric, na.rm = TRUE)

# Print the result
print(average_value)


# 1. Create a vector of column names from logit_diff_llava that match the head-layer combinations in fact_heads
head_layer_combinations <- with(cfact_heads, paste("logit_lens_head_out_L", layer, "H", head, sep = ""))

# 2. Subset logit_diff_llava to keep only the relevant columns
logit_subset <- logit_diff_llava[ , colnames(logit_diff_llava) %in% head_layer_combinations]

# 3. Convert the subset to numeric (if not already numeric) and flatten to a vector
logit_subset_numeric <- as.numeric(unlist(logit_subset))

# 4. Compute the average of the selected columns
average_value <- mean(logit_subset_numeric, na.rm = TRUE)

# Print the result
print(average_value)





###########################################################################################################################
###########################################################################################################################
###                                                                                                                     ###
###                                                                                                                     ###
###########################################################################################################################
###########################################################################################################################


# ggplot a lineplot
df_multi_k <- multi_k_llava %>%
  rename(image_diff = `Image Cfact>Fact`)


# apply 100-image_diff
df_multi_k <- df_multi_k %>%
  mutate(image_diff = 100-image_diff)

# 3. Extract the baseline (no intervention) value
baseline_value <- df_multi_k %>%
  filter(ExperimentDesc == "Baseline (no intervention)") %>%
  pull(image_diff) %>%
  unique()





df_filtered$`Image Pos Higher`


library(ggplot2)

# Assuming df_multi_k contains columns: lambda, image_diff, and k_heads

# Example plot:
library(ggplot2)

# Filter data where lambda == 2
df_filtered <- df_multi_k %>%
  filter(lambda == 2)
df_filtered$`Image Pos Higher`

# Plot the profile for different levels of k_heads
ggplot(df_filtered, aes(x = as.factor(k_heads), y = `Image Pos Higher` , group = 1)) + 
  geom_line(size = 1, color = "#5F9ED1") +  # Line connecting points
  geom_point(size = 3, color = "#5F9ED1") +  # Add points
  labs(
    x = "N. Heads",                  # Label for x-axis
    #y = "Factual Accuracy (%)",                 # Label for y-axis
    y = "Max Position",
    title = ""
  ) +
  theme_custom()



ggsave("plots/v1/llava_experiment1_multik_lambda2_maxPos.pdf", width = 6, height = 4, dpi = 300)






#########################################################
# Load necessary library
library(dplyr)
library(ggplot2)

# Filter for position 12 and labels containing 'attn_out' and 'mlp_out'
attn_out_df <- logit_attribution_data %>%
  filter(position == 12 & grepl("attn_out", label, ignore.case = TRUE))

mlp_out_df <- logit_attribution_data %>%
  filter(position == 12 & grepl("mlp_out", label, ignore.case = TRUE))


# plot Attention
ggplot(mlp_out_df, aes(
  x    = factor(c(0:11)),       # treat layer as categorical
  y    = diff_mean,
)) +
  geom_col(
    position = position_dodge(width = 0.8),
    width    = 0.7, fill="#CC6677"
  ) +
  labs(
    x = "Layer",
    y = "Logit Diff",
    title = "MLP"
  ) +
#  scale_y_continuous(
#    name = "Factual Prevalence",
#    breaks = c(-25,-15,0,15,25),
#    labels = c("Counter-\nfactual", "-15%", "0%", "15%", "Factual"),
#    limits = c(-25,25)
#  )+
  #scale_x_discrete(
   # name = "Layer",
    #breaks=c(0:12),
    #breaks = c(0:31),
    #labels = c("0","", "","", "4","", "","", "8","",  "","",  "12","",  "","",  "16","",  "","",  "20","",  "","",  "24","",  "","",  "28","",  "","","32","","","","36","", "","",  "40","",  "","",  "44","",  "","",  "48"),                                                
    #labels = c("0", "","2","","4", "","6","", "8","",  "10","",  "12","",  "14","",  "16","",  "18","",  "20","",  "22","",  "24","",  "26","",  "28","",  "30",""),
  #)+
  
  coord_cartesian(ylim = c(-1.5, 1.5)) +
  theme_custom() +
  theme(
    legend.position = "bottom"
  )





mlp_bar <-
