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
df_exp1_llava <-  read_csv("//wsl.localhost/Ubuntu/home/francesco/VisualComp/results/1_heads_ablation/v16_arXiv/llava-hf-llava-v1.6-mistral-7b-hf_2025-07-10_15-57-10/v16_arXiv.csv")
df_exp1_gemma <-  read_csv("//wsl.localhost/Ubuntu/home/francesco/VisualComp/results/1_heads_ablation/v16_arXiv/google-gemma-3-12b-it_2025-07-10_16-38-30/v16_arXiv.csv")

#control
df_exp1_llava <- read_csv("//wsl.localhost/Ubuntu/home/francesco/VisualComp/results/1_heads_ablation/full_experiment_llava-hf-llava-v1.6-mistral-7b-hf_2025-05-16_10-59-20v11_control/full_experiment_llava-hf-llava-v1.6-mistral-7b-hf_2025-05-16_10-59-20v11_control.csv")
df_exp1_gemma <- read_csv("//wsl.localhost/Ubuntu/home/francesco/VisualComp/results/1_heads_ablation/full_experiment_google-gemma-3-12b-it_2025-05-18_22-26-46v11_control/full_experiment_google-gemma-3-12b-it_2025-05-18_22-26-46v11_control.csv")
View(full_experiment_llava_hf_llava_v1_6_mistral_7b_hf_2025_05_10_15_59_03v9)



# ggplot a lineplot
df_exp1_llava <- df_exp1_llava %>%
  rename(image_diff = `Image Cfact>Fact`)

# ggplot a lineplot
df_exp1_gemma <- df_exp1_gemma %>%
  rename(image_diff = `Image Cfact>Fact`)

# apply 100-image_diff
df_exp1_llava <- df_exp1_llava %>%
  mutate(image_diff = 100-image_diff)

df_exp1_gemma <- df_exp1_gemma %>%
  mutate(image_diff = 100-image_diff)

# 3. Extract the baseline (no intervention) value
baseline_value <- df_exp1_llava %>%
  filter(Lambda == 0) %>%
  pull(image_diff) %>%
  unique()

baseline_value_gemma <- df_exp1_gemma %>%
  filter(Lambda == 0) %>%
  pull(image_diff) %>%
  unique()

df_exp1_llava <- df_exp1_llava %>% filter(Lambda >= -3 & Lambda <= 3)
df_exp1_gemma <- df_exp1_gemma %>% filter(Lambda >= -3 & Lambda <= 3)


# 4. Plot
# install.packages("latex2exp")   # if you haven't already
library(ggplot2)
library(latex2exp)

ggplot(df_exp1_llava, aes(x = Lambda, y = image_diff)) +
  geom_hline(
    yintercept = baseline_value,
    linetype   = "dashed",
    size       = 1,
    col="darkgrey"
  ) +
  
  geom_line(size = 1.5, col="#009E73") +
  geom_point(size = 3, col="#009E73") +
  labs(
    x = "Enhance Counterfactual Heads                Enahance Factual Heads",                           # LaTeX Gamma
    y = "Factual Accuracy (%)"               # escaped percent
  ) +
  theme_custom() 

# Combined plot for llava and Gemma

combined_data <- rbind(
  df_exp1_llava %>% mutate(Model = "LLaVA-NeXT"),
  df_exp1_gemma %>% mutate(Model = "Gemma3")
)

ggplot(combined_data, aes(x = Lambda, y = image_diff, color = Model)) +
  geom_hline(
    yintercept = baseline_value_gemma,
    linetype   = "dashed",
    size       = 1,
    col="#009E73"
  ) +
  geom_hline(
    yintercept = baseline_value ,
    linetype   = "dashed",
    size       = 1,
    col="#D55E00"
  ) +
  
  geom_line(size = 1.5) +
  geom_point(size = 3) +
  ylim(15,85)+
  labs(
    x = expression(lambda),                           # LaTeX Gamma
    y = "Factual Accuracy (%)"               # escaped percent
  ) +
  scale_color_manual(values=c("#009E73", "#D55E00")) + # Custom colors for the lines
  theme_custom()



ggsave(
  filename = "plots/v2/experiment1.pdf",
  dpi = 400, width = 6, height = 4, units =  'in', scale = 1
)

