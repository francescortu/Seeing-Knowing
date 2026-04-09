library(extrafont)

p1 <- c("control"="#4477AA", 
        "heads"="#CCBB44", 
        "grad"="#AA3377")

p2 <- c("EOI"="#77AADD", 
        "Internal image"="#BBCC33", 
        "Last image"="#EE8866")

p3 <- c("EOI"="#44AA99", 
        "Internal image"="#CC6677", 
        "Last image"="#999933")

p  <- c("EOL"= "#0072B2",
        "Internal image"="#D55E00",
        "Last image"="#009E73",
        "EOI"="#CC79A7",
        "0,1,32"="#999999")


p  <- c("EOL"= "#DDDDDD",
        "Internal image"="#EE8866",
        "Last image"="#EEDD88",
        "EOI"="#77AADD")

p <- c("EOL"= "#FFC107",
       "Internal image"="#D55E00",#EE8866
       "Last image"='#5F9ED1',
       "EOI"="#0C8A68")


theme_custom <- function() {
  theme_bw(base_size = 13) + # Start with a minimal theme
    theme(
      # Background color
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      
      # Title and axis labels
      plot.title = element_text(size = 15, color = "#333333"), #hjust = 0.5),
      plot.subtitle = element_text(size = 13, color = "#444444"),
      axis.title = element_text(size = 15, color = "#444444"),
      axis.text = element_text(size = 14, color = "#555555"),
      
      # Grid lines
      panel.grid.major = element_line(color = "grey90", size = 0.5),
      panel.grid.minor = element_line(color = "grey90", size = 0.5),
      
      # Legend
      legend.background = element_rect(fill = "white", color = "NA"),
      legend.text = element_text(size = 14, color = "#444444"),
      legend.title = element_text(size = 14, color = "#444444"),
      legend.position = 'bottom',
      legend.box.spacing = unit(-0.5, "pt"),
      
      # Margins and padding
      plot.margin = margin(10, 10, 10, 10),
      
      # Optional: Customize axis lines or ticks
      axis.line = element_line(color = "grey70"),
      axis.ticks = element_line(color = "grey70")
    )
}
