# ──────────────────────────────────────────────────────────────
# Stats - analyzing accuracy to comprehension Qs
# in privative and subsective, concrete and abstract phrases
# Author: Ryan Law
# ──────────────────────────────────────────────────────────────

# ---- Setup ----
# Load required libraries
# library(readr)
library(lme4)
library(lmerTest)
library(emmeans)
library(tibble)
library(dplyr)
library(MuMIn)
library(car)
# library(effects)

# ---- Configuration ----
# base_dir <- "/path/to/your/project/fake_diamond" # e.g., /home/rl05/projects/fake_diamond
data_dir <- '/imaging/hauk/rl05/fake_diamond/data/logs'
results_dir <- '/imaging/hauk/rl05/fake_diamond/results/behavioral'
analysis_dir <- '/imaging/hauk/rl05/fake_diamond/scripts/analysis/behavioural'
group_data_path <- file.path(analysis_dir, "group_data.csv")
models_dir <- file.path(analysis_dir, 'models')
analysis <- 'Accuracy'

# Create the results directory if it doesn't exist
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}
if (!dir.exists(models_dir)) {
  dir.create(models_dir, recursive = TRUE)
}

cat("--- Reading and preparing data ---\n")
group_data <- readr::read_csv(group_data_path)
# Set data types for factors
group_data <- mutate(group_data,
                     participant = factor(participant),
                     item_nr = factor(item_nr),
                     composition = factor(composition),
                     denotation = factor(denotation),
                     concreteness = factor(concreteness),
                     condition = factor(condition))
group_data <- as_tibble(group_data)

cat("Data summary:\n")
print(group_data)


cat("\n--- Analyzing group-level accuracy ---\n")

# ---- GLMER control ----
control_glmer <- glmerControl(
  optimizer = "bobyqa",          # More robust than default "Nelder_Mead"
  optCtrl = list(maxfun = 2e5)   # Increase max function evaluations to help convergence
)

# ---- Fit LMER for RT ----
model_path_acc <- file.path(models_dir, "final_model_acc.rds")

if (!file.exists(model_path_acc)) {
  cat("\n--- Fitting GLMER model for accuracy ---\n")
  cat("Fitting maximal model: (concreteness * denotation | participant) + (1 | item_nr)\n")

  m_max_acc <- glmer(hit ~ concreteness * denotation +
                      (concreteness * denotation | participant) + (1 | item_nr),
                    data = group_data, family = 'binomial',
                    control = control_glmer)

  if (isSingular(m_max_acc)) {
    cat("Maximal model is singular. Trying zero-correlation slopes...\n")

    m_zc_acc <- glmer(hit ~ concreteness * denotation +
                        (concreteness * denotation || participant) + (1 | item_nr),
                      data = group_data, family = 'binomial',
                      control = control_glmer)

    if (isSingular(m_zc_acc)) {
      cat("Still singular. Trying removing interaction term...\n")

      m_step1_acc <- glmer(hit ~ concreteness * denotation +
                            (concreteness + denotation || participant) + (1 | item_nr),
                          data = group_data, family = 'binomial',
                          control = control_glmer)

      if (isSingular(m_step1_acc)) {
        cat("Still singular. Removing concreteness ...\n")

        m_step2_acc <- glmer(hit ~ concreteness * denotation +
                              (denotation || participant) + (1 | item_nr),
                            data = group_data, family = 'binomial',
                            control = control_glmer)

        if (isSingular(m_step2_acc)) {
          cat("Still singular. Trying only by-participant varying intercepts...\n")
          m1_acc <- glmer(hit ~ concreteness * denotation +
                            (1 | participant) + (1 | item_nr),
                          data = group_data, family = 'binomial',
                          control = control_glmer)

          final_model_acc <- m1_acc

          if (isSingular(final_model_acc)) {
            warning("Even the random-intercepts-only model is singular.")
          }
        } else {
          final_model_acc <- m_step2_acc
        }
      } else {
        final_model_acc <- m_step1_acc
      }
    } else {
      final_model_acc <- m_zc_acc
    }
  } else {
    final_model_acc <- m_max_acc
  }

  # Announce the final model that will be used for analysis.
  cat("\n--- Final Accuracy Model Chosen ---\n")
  print(formula(final_model_acc))
  saveRDS(final_model_acc, file = "models/final_model_acc.rds")
} else {
  cat("\n--- Loading saved final Acc model ---\n")
  final_model_acc <- readRDS(model_path_acc)
  print(formula(final_model_acc))
}

# ---- Summarise the final model ----
summary_anova <- Anova(final_model_acc)
r_squared <- r.squaredGLMM(final_model_acc)

cat("\n#### ANOVA Results\n")
print(summary_anova)

cat("\n#### R-squared (Nakagawa & Schielzeth)\n")
print(r_squared)

# ---- Post-hoc Comparisons: Interaction IS Significant ----
emm_interaction <- emmeans(final_model_acc, ~ concreteness * denotation)
em_deno <- emmeans(final_model_acc, ~ denotation) # main effect of denotation only

# Back-transform EMMs
emm_interaction_response <- summary(emm_interaction, type = "response") 
em_deno_resp <- summary(em_deno, type = "response")   

# Pairwise post-hoc comparisons
# main effect of denotation
pairwise_denotation <- contrast(em_deno, method = "pairwise")
pairwise_denotation_response <- summary(pairwise_denotation, infer = TRUE, type = "response")

# Simple effect of CONCRETENESS within each DENOTATION level
pairwise_concreteness_by_denotation <- contrast(emm_interaction, 
                                                method = "pairwise", 
                                                by = "denotation")

# Simple effect of DENOTATION within each CONCRETENESS level
pairwise_denotation_by_concreteness <- contrast(emm_interaction, 
                                                method = "pairwise", 
                                                by = "concreteness")

cat("\n#### Post-hoc: Effect of DENOTATION\n")
print(pairwise_denotation_response)

cat("\n#### Post-hoc: Effect of CONCRETENESS within each DENOTATION level\n")
print(pairwise_concreteness_by_denotation)

cat("\n#### Post-hoc: Effect of DENOTATION within each CONCRETENESS level\n")
print(pairwise_denotation_by_concreteness)

# ---- Aggregate Results ----
result_text <- c(
  paste("\n\n############################################################"),
  paste("########## RESULTS FOR:", analysis),
  paste("############################################################\n"),
  "\n----- ANOVA Results (Type III with Satterthwaite's method) -----\n",
  capture.output(print(summary_anova)),
  "\n----- R-squared (Nakagawa & Schielzeth) -----\n",
  capture.output(print(r_squared)),
  "\n----- Estimated Marginal Means of each condition -----\n",
  capture.output(print(emm_interaction_response)),
  "\n----- EMMs for DENOTATION -----\n",
  capture.output(print(em_deno_resp)),
  "\n----- Pairwise CONTRASTS for DENOTATION -----\n",
  capture.output(print(pairwise_denotation_response)),
  "\n----- Post-hoc: Effect of CONCRETENESS within each DENOTATION -----\n",
  capture.output(print(pairwise_concreteness_by_denotation)),
  "\n----- Post-hoc: Effect of DENOTATION within each CONCRETENESS -----\n",
  capture.output(print(pairwise_denotation_by_concreteness))
)

# ---- Write summary output to file ----
summary_file <- file.path(results_dir, paste0("stats_", analysis, ".txt"))
writeLines(result_text, con = summary_file)

cat("\n\nAll done. Summary written to:\n", summary_file, "\n")
