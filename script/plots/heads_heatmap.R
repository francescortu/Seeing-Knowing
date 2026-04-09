# install.packages(c("readr", "tidyr", "ggplot2"))  # if you haven't already

library(readr)
library(tidyr)
library(ggplot2)
library(dplyr)
library(khroma)
library(rjson)





#################################################################################
bright <- color("bright")
# 1. Read in the CSV
#    Adjust the path if your file lives elsewhere
## lava
df <- read_csv("results/0_heads_selection/logit_lens_2025-05-09_15-10-32/selected_heads.csv")
stats <- fromJSON(file="results/0_heads_selection/logit_lens_2025-05-09_15-10-32/stats.json")

## gemma
df <- read_csv("//wsl.localhost/Ubuntu/home/francesco/VisualComp/results/0_heads_selection/logit_lens_2025-05-14_15-20-34/selected_heads.csv")
stats <- fromJSON(file="//wsl.localhost/Ubuntu/home/francesco/VisualComp/results/0_heads_selection/logit_lens_2025-05-14_15-20-34/stats.json")

df2 <- df %>%
  mutate(
    # grab the digits after 'L' and before 'H'
    layer = as.integer(gsub(".*L(\\d+)H.*", "\\1", Head)),
    # grab the digits after the final 'H'
    head  = as.integer(gsub(".*H(\\d+)$",   "\\1", Head))
  )

# 3. Turn layer/head into factors so axes sort naturally
df2 <- df2 %>%
  mutate(
    layer = factor(layer, levels = sort(unique(layer))),
    head  = factor(head,  levels = sort(unique(head)))
  )

# sum +0.5 to all the Values to make them positive
df2 <- df2 %>%
  mutate(
    Value = Value + 0.5
  )

#multiply for 100
df2 <- df2 %>%
  mutate(
    Value = 100 - (Value * 100)
  )


df2 <- df2 %>%
  mutate(
    Value = (Value -50)
  )


AXIS_TEXT_SIZE <- 22
AXIS_TITLE_SIZE <- 28



# after your existing code, add:

library(patchwork)


# 1. Build a summary data.frame from stats
summary_df <- tibble(
  metric = factor(
    c("Cfact", "Fact", "All"),
    levels = c("Cfact", "Fact", "All")
  ),
  mean   = c(
    as.numeric(stats$mean_values_cfact),
    as.numeric(stats$mean_values_fact),
    as.numeric(stats$mean_all_matrix)
  ),
  se     = c(
    as.numeric(stats$std_err_cfact),
    as.numeric(stats$std_err_fact),
    as.numeric(stats$std_err_all_matrix)
  )
)


df2_with_attn <- df2 %>%
  mutate(
    layer = as.numeric(as.character(layer)),
    head = as.numeric(as.character(head))
  ) %>%
  left_join(
      full_attn_to_img,
  by = c("layer", "head")
)

fact_heads <- df2_with_attn %>%
  filter(
    Value > 24.5
  )
cfact_heads <- df2_with_attn %>%
  filter(
    Value < -24.5
  )

df2_with_attn %>%   summarise(
  q25   = quantile(Value, 0.02, na.rm = TRUE),
  q75   = quantile(Value, 0.98, na.rm = TRUE),
  n25   = sum(Value <= q25, na.rm = TRUE),
  n75   = sum(Value >= q75, na.rm = TRUE)
)

mean(fact_heads$value)
sd(fact_heads$value)
mean(cfact_heads$value)
sd(cfact_heads$value)
# create the dataframe
summary_df <- data.frame(
  metric = c("Counterfactual", "Factual", "All"),
  mean = c(mean(cfact_heads$value), mean(fact_heads$value), mean(df2_with_attn$value)),
  se = c(sd(cfact_heads$value)/sqrt(length(cfact_heads$value)), sd(fact_heads$value)/sqrt(length(fact_heads$value)), sd(df2_with_attn$value)/sqrt(length(df2_with_attn$value)))
)

# 2. Bar plot with error bars
bar_plot <- ggplot(summary_df, aes(x = metric, y = mean, fill = metric)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.2) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_fill_manual(values = c(
    "Counterfactual" = "#DA1E28",  # a strong red
    "Factual"     = "#0072C3",  # a strong blue
    "All"  = "darkgrey"   # a strong yellow
  )) +
  labs(x = NULL, y = "% Attention to Image \n") +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(size=AXIS_TEXT_SIZE, angle = 45, hjust = 1),
    axis.text.y = element_text(size=AXIS_TEXT_SIZE),
    axis.title.y = element_text(size = AXIS_TITLE_SIZE),
    panel.grid.major.x = element_blank(),
    )

# 3. Combine heatmap and bar plot
lim <- max(abs(df2$Value), na.rm = TRUE)

combined <-free(ggplot(df2, aes(x = layer, y = head, fill = Value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(
    low      = "#da1e28",
    mid      = "white",
    high     = "#0072c3",
    midpoint = 0,
    limits   = c(-lim, lim),
    breaks =  c(-lim, -25, 0,25, lim),
    labels = c(
     "Counter-\nfactual",
     "25",
      "50",
      "75",
      "Factual"
    ),
    name     = "Factual Accuracy (%)",
    guide    = guide_colorbar(
      title.position = "top",
      title.hjust    = 0.5,
      barwidth       = unit(8, "cm"),
    )
  ) +
  scale_y_discrete(
    name = "Head",
    breaks = c(0:31),
    labels = c("0","", "2","", "4","", "6","", "8","",  "10","",  "12","",  "14","",  "16","",  "18","",  "20","",  "22","",  "24","",  "26","",  "28","",  "30",""),
  )+
  scale_x_discrete(
    name = "Layer",
    breaks = c(0:31),
    labels = c("0","", "2","", "4","", "6","", "8","",  "10","",  "12","",  "14","",  "16","",  "18","",  "20","",  "22","",  "24","",  "26","",  "28","",  "30",""),
  )+
  labs(x = "Layer", y = "Head") +
  theme_custom() +
  theme(
    panel.grid = element_blank(),
    legend.position = "bottom",
    axis.text  = element_text(size = 22),
    axis.title = element_text(size = 28),
    legend.title = element_text(size = 22),
    legend.text  = element_text(size = 20),
    legend.key.size = unit(1, "cm"),
    legend.margin = margin(t = 3)
  ) )+plot_spacer()+ bar_plot + 
  plot_layout(widths = c(4,0.2, 0.8))

combined

# 4. Save
ggsave("plots/v1/heads_heatmap_with_stats.pdf", combined, 
       width = 17, height = 9, dpi = 300)

#####################
logit_lens_mlp$perc <- 100*(-logit_lens_mlp$Value )
logit_lens_attn$perc <- 100*(-logit_lens_attn$Value )

attn_df <- logit_lens_attn %>%
  mutate(
    component = "Attention",
    layer     = as.integer(gsub(".*out_(\\d+)$", "\\1", Type))
  )

mlp_df <- logit_lens_mlp %>%
  mutate(
    component = "MLP",
    layer     = as.integer(gsub(".*out_(\\d+)$", "\\1", Type))
  )

# 2. Combine
df_combined <- bind_rows(attn_df, mlp_df)

# plot Attention
ggplot(df_combined[df_combined$component == "Attention",], aes(
  x    = factor(layer),       # treat layer as categorical
  y    = perc,
  fill = component
)) +
  geom_col(
    position = position_dodge(width = 0.8),
    width    = 0.7, fill="#CCBB44"
  ) +
  labs(
    x = "Layer",
    y = "Factual Prevalence",
    title = "Attention"
  ) +
  scale_y_continuous(
    name = "Factual Prevalence",
    breaks = c(-25,-15,0,15,25),
    labels = c("Counter-\nfactual", "-15%", "0%", "15%", "Factual"),
    limits = c(-25,25)
  )+
  scale_x_discrete(
    name = "Layer",
    breaks=c(0:48),
    #breaks = c(0:31),
    labels = c("0","", "","", "4","", "","", "8","",  "","",  "12","",  "","",  "16","",  "","",  "20","",  "","",  "24","",  "","",  "28","",  "","","32","","","","36","", "","",  "40","",  "","",  "44","",  "","",  "48"),                                                
    #labels = c("0", "","2","","4", "","6","", "8","",  "10","",  "12","",  "14","",  "16","",  "18","",  "20","",  "22","",  "24","",  "26","",  "28","",  "30",""),
  )+
  
  coord_cartesian(ylim = c(-25, 25)) +
  theme_custom() +
  theme(
    legend.position = "bottom"
  )
ggsave("plots/v1/gemma_attn_barplot.pdf", width=5.5, height=3, dpi=400)


ggplot(df_combined %>% filter(component == "MLP"), aes(
  x    = factor(layer),       # treat layer as categorical
  y    =  perc,
  fill = component
)) +
  geom_col(
    position = position_dodge(width = 0.8),
    width    = 0.7, fill="#CC6677"
  ) +
  scale_fill_manual(
    name   = "Model Component:",
    values = "#CC6677",   # unnamed, so applied to the single level
    labels = "MLP",
    guide  = guide_legend(order = 1)
  ) +
  labs(
    x = "Layer",
    y = "Factual Prevalence",
    title = "MLP"
  ) +
  scale_y_continuous(
    name = "Factual Prevalence",
    breaks = c(-25,-15,0,15,25),
    labels = c("Counter-\nfactual", "-15%", "0%", "15%", "Factual"),
    limits = c(-25,25)
  )+
  scale_x_discrete(
    name = "Layer",
    breaks=c(0:48),
    #breaks = c(0:31),
    labels = c("0","", "","", "4","", "","", "8","",  "","",  "12","",  "","",  "16","",  "","",  "20","",  "","",  "24","",  "","",  "28","",  "","","32","","","","36","", "","",  "40","",  "","",  "44","",  "","",  "48"),                                                
    #labels = c("0", "","2","","4", "","6","", "8","",  "10","",  "12","",  "14","",  "16","",  "18","",  "20","",  "22","",  "24","",  "26","",  "28","",  "30",""),
    
    )+
  
  coord_cartesian(ylim = c(-25, 25)) +
  theme_custom() +
  theme(
    legend.position = "bottom",
    #axis.title.y = element_blank(),
  )

# 3. Plot

ggsave("plots/v1/gemma_mlp_barplot.pdf", width=5.5, height=3, dpi=400)


component_colors <- c(
  Attention = "#CCBB44",
  MLP       = "#CC6677"
)

ggplot(df_combined, aes(
  x    = factor(layer),
  y    = perc,
  fill = component
)) +
  geom_col(
    position = position_dodge(width = 0.8),
    width    = 0.7
  ) +
  scale_fill_manual(
    name   = "Model Component:",
    values = component_colors,
    labels = c("Attention", "MLP")
  ) +
  scale_y_continuous(
    name   = "Preference Score",
    breaks = c(-25, -15, 0, 15, 25),
    limits = c(-25, 25),
    labels = c(
      "Visual\nContext",  "-15", "0",
      "15",  "Inner\nKnowledge"
    )
  ) +
  labs(
    x     = "Layer",
    title = NULL
  ) +
  theme_custom() +
  theme(
    legend.position = "bottom"
  )
ggsave("plots/v1/combined_barplot.pdf", width=11, height=3, dpi=400)
