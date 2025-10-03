# ──────────────────────────────────────────────────────────────
# Stats - decoding concreteness in early and late time windows 
# in privative and subsective phrases
# Author: Ryan Law
# ──────────────────────────────────────────────────────────────

# ---- Setup ----
# Load required libraries
library(lme4)
library(lmerTest)
library(emmeans)
library(MuMIn)
library(dplyr)
library(ggplot2)
library(tibble)

# ---- Configuration ----
analysis <- 'concreteness_xcond'
window_type <- 'single'
sfreq_val <- 100
rois <- c('anteriortemporal-lh', 'temporoparietal-lh')
results <- list()
output_dir <- '/imaging/hauk/rl05/fake_diamond/results/neural/decoding/stats_interaction'

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# ---- Main Analysis Loop ----
for (roi in rois) {
  
    identifier <- paste(analysis, window_type, roi, sfreq_val, sep = "_")
    cat(paste("\n\n### Analysis for ROI:", roi, "\n"))

    input_dir <- file.path('/imaging/hauk/rl05/fake_diamond/figures/decoding/', analysis, 'diagonal/logistic/ROI', window_type, 'micro_ave')
    input_file_path <- file.path(input_dir, paste0('scores_timewindow-averaged_', roi, '_', sfreq_val, 'Hz.csv'))

    if (!file.exists(input_file_path)) {
    warning(paste("Data file not found, skipping:", input_file_path))
    next
    }

    # Load and prepare data
    data <- read.csv(input_file_path) |> 
    as_tibble() |> 
    mutate(
        timewindow = factor(timewindow),
        test_on = factor(test_on),
        subject = factor(subject)
    )

    # Fit model
    m_max <- lmer(score ~ timewindow * test_on + 
                    (timewindow + test_on | subject), # no interaction term in the random-effects structure, otherwise overly complex to begin with
                data)

    if (isSingular(m_max)) {
        # 2. drop correlations
        m_zc <- lmer(score ~ timewindow * test_on + 
                        (timewindow * test_on || subject),
                    data)

        if (isSingular(m_zc)) {
            # 3. inspect PCA
            print(rePCA(m_zc))
            # Suppose test_on slope ~0 → remove it
            m1 <- lmer(score ~ timewindow * test_on +
                        (timewindow + 1 || subject),
                    data)
            
            if (isSingular(m1)) {
            # Suppose timewindow slope ~0 → remove it too
            m2 <- lmer(score ~ timewindow * test_on +
                        (1 | subject),
                        data)
            final_model <- m2
            } else {
            final_model <- m1
            }
        } else {
            final_model <- m_zc
        }
    } else {
    final_model <- m_max
    }

    # final_model is now parsimonious yet still justified by your design.
    print(formula(final_model))
    summary(final_model)

    # Summarise model
    summary_anova <- anova(final_model)
    r_squared <- r.squaredGLMM(final_model)
    cat("\n#### ANOVA Results\n")
    print(summary_anova)
    cat("\n#### R-squared (Nakagawa & Schielzeth)\n")
    print(r_squared)

    # Since the interaction is not significant, we look at the main effects.
    # 1. Post-hoc comparisons for the main effect of 'timewindow'
    emm_timewindow <- emmeans(final_model, ~ timewindow)
    pairwise_timewindow <- pairs(emm_timewindow)

    # 2. Post-hoc comparisons for the main effect of 'test_on'
    emm_test_on <- emmeans(final_model, ~ test_on)
    pairwise_test_on <- pairs(emm_test_on)

    cat("\n#### Post-hoc for Main Effect of 'timewindow'\n")
    print(pairwise_timewindow)

    cat("\n#### Post-hoc for Main Effect of 'test_on'\n")
    print(pairwise_test_on)


    # # Plot
    # plot_title <- paste("Pairwise Comparisons for", identifier)
    # emm_plot <- plot(emm, comparisons = TRUE, horizontal = FALSE) +
    # labs(
    #     title = plot_title,
    #     subtitle = "Estimated Marginal Means with 95% CIs",
    #     x = "Time Window",
    #     y = "Estimated Marginal Mean of Score"
    # ) +
    # theme_bw(base_size = 14) +
    # theme(axis.text.x = element_text(angle = 45, hjust = 1))

    # # Save plot
    # plot_file_path <- file.path(output_dir, paste0('pairwise_plot_', identifier, '.png'))
    # ggsave(plot_file_path, plot = emm_plot, width = 12, height = 9, dpi = 300, device = 'png')

    # Aggregate output for summary
    result_text <- c(
        paste("############################################################"),
        paste("########## RESULTS FOR:", identifier),
        paste("############################################################\n"),
        "\n----- Chosen model -----\n",
        capture.output(print(formula(final_model))),
        "\n----- ANOVA Results (Type III with Satterthwaite's method) -----\n",
        capture.output(print(summary_anova)),
        "\n----- R-squared (Nakagawa & Schielzeth) -----\n",
        capture.output(print(r_squared)),
        "\n----- Post-hoc for Main Effect of 'timewindow' -----\n",
        capture.output(print(pairwise_timewindow)),
        "\n----- Post-hoc for Main Effect of 'test_on' -----\n",
        capture.output(print(pairwise_test_on))
    )
    results[[identifier]] <- result_text

    # ---- Write summary output to file ----
    summary_file <- file.path(output_dir, paste0("summary_stats_", analysis, "_", roi, ".txt"))
    writeLines(result_text, con = summary_file)

    cat("\n\nAll done. Summary written to:\n", summary_file, "\n")
}

