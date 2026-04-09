library(readr)
library(tidyr)
library(ggplot2)
library(dplyr)
library(khroma)
library(rjson)
library(patchwork)
library(dplyr)

library(reticulate)
library(readr)
# llava 
df <- read_csv("//wsl.localhost/Ubuntu/home/francesco/VisualComp/results/2_ImgCfactLocalization/v16_arXiv/llava-hf-llava-v1.6-mistral-7b-hf_2025-07-07_17-25-14/results.csv") #read_csv("//wsl.localhost/Ubuntu/home/francesco/VisualComp/results/2_ImgCfactLocalization/ImgCfactLoc_llava-hf-llava-v1.6-mistral-7b-hf_2025-05-12_20-19-18v5/results.csv")
# gemma
df <- read_csv("//wsl.localhost/Ubuntu/home/francesco/VisualComp/results/2_ImgCfactLocalization/v16_arXiv/google-gemma-3-12b-it_2025-07-03_18-56-49/results.csv")

library(dplyr)
library(ggplot2)

# 1. Filter out baseline and get full-pixel counts per condition
full_pixels <- df %>%
  filter(ExperimentDesc != "baseline", threshold == 0) %>%
  select(ExperimentDesc, full_pixels = avg_num_pixel)

# 2. Join and compute percentage removed
df_plot <- df %>%
  filter(ExperimentDesc != "baseline") %>%
  left_join(full_pixels, by = "ExperimentDesc") #%>%
 # mutate(
  #  pct_removed = avg_num_pixel / full_pixels * 100
  #)

# 3. Make it a factor so your legend stays in a sensible order
df_plot <- df_plot %>%
  mutate(
    ExperimentDesc = factor(ExperimentDesc,
                            levels = unique(ExperimentDesc)
    )
  )

# 3.5 Chage Image Cfact>Fact to Value equal to -Image Cfact>Fact
df_plot <- df_plot %>%
  mutate(
    `Image Cfact>Fact` = 100 - `Image Cfact>Fact`
  )


# multiply by 100 threshold
df_plot <- df_plot %>%
  mutate(
    threshold = threshold * 100
  )

# add a thresold = 0 for all the ExperimentDesc using the value in experimentDesc == Baseline

# Extract the baseline value for Image Cfact>Fact
baseline_value <- df %>%
  filter(ExperimentDesc == "baseline") %>%
  pull(`Image Cfact>Fact`)
baseline_value <- 100 - baseline_value # Convert to percentage removed
# Define the ablation types
ablation_types <- c("resid_ablation", "resid_ablation_control", "resid_ablation_grad")

# Get column names from df_plot
col_names <- names(df_plot)

# Create new rows as a tibble
new_rows <- tibble::tibble(
  ExperimentDesc = ablation_types,
  `% top pixel` = "N/A",
  threshold = 0,
  LogitDiff = NA_real_,
  avg_num_pixel = NA_real_,
  `Image Cfact logit` = NA_real_,
  `Image Fact Logit` = NA_real_,
  `Text Cfact Logit` = NA_real_,
  `Text Fact Logit` = NA_real_,
  `Image Cfact>Fact` = baseline_value
  # Add NA for other columns as necessary
)
missing_cols <- setdiff(col_names, names(new_rows))
for (col in missing_cols) {
  new_rows[[col]] <- NA
}
new_rows <- new_rows[, col_names] # Ensure correct column order

# Bind the new rows to df_plot
df_plot <- dplyr::bind_rows(df_plot, new_rows)

ggplot(df_plot, aes(
  x        = threshold,
  y        = `Image Cfact>Fact`,
  color    = ExperimentDesc,
  linetype = ExperimentDesc
)) +
  geom_line(size = 1.5) +
  geom_point(size = 3) +
  scale_x_continuous(
    name   = "% Pixels Removed",
    limits = c(0, 100),
    breaks = seq(0, 100, by = 20)
  ) +
  scale_y_continuous(name = "Factual Accuracy (%)") +
  scale_color_manual(
    name   = "",
    values = c("#009E73", "darkgray", "#D55E00"),
    labels = c("Through Attn Heads","Random", "Through Gradients" )
  ) +
  scale_linetype_manual(
    name   = "",
    values = c("solid", "dotted", "solid"),
    labels = c("Through Attn Heads", "Random", "Through Gradients")
  ) +
  labs(
    title="Gemma3"
    #title="LLaVA-NeXT"
  )+
  guides(
    color    = guide_legend(
      override.aes = list(
        linetype = c("solid", "dotted", "solid")
      )
    ),
    linetype = guide_legend(
      override.aes = list(
        colour   = c("#009E73", "darkgray", "#D55E00")
      )
    )
  ) +
  theme_custom() +
  theme(legend.position = "bottom")

ggsave(
  filename = "plots/v2/llava_experiment2.pdf",
  dpi = 400, width = 6, height = 4, units =  'in', scale = 1
)
ggsave(
  filename = "plots/v2/gemma_experiment2.pdf",
  dpi = 400, width = 6, height = 4, units =  'in', scale = 1
)
