#### 00 load librarys ####
library(tidyverse)
library(hrbrthemes)
library(tidymodels)
library(ggsci)
library(reshape2)
library(themis)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#### 01 Explore Data ####
load("forrest.Rdata")

set.seed(51)
forrest %>%
  filter(!is.na(isforrest)) %>%
  group_by(isforrest) %>%
  sample_n(4) %>%
  select(isforrest, text)

# at first, we have to fix missing values in "duration"
sum(is.na(forrest$duration))
forrest$duration <- ((forrest$offset - forrest$onset) / 1000)
# it seems that all other "sentences" with duration > 13.XX aren't real sentences
forrest[forrest$duration > 10,c("duration","isforrest","text")] %>% filter(isforrest == "OTHER")

forrest %>%
  filter(duration < max(forrest[forrest$duration > 10,c("duration","isforrest","text")] %>% filter(isforrest == "FORREST") %>% select(duration))) %>%
  mutate(sent_mean = sent_sum / word_count,
         comp_mean = comp_score / word_count,
         open_mean = open / word_count,
         closed_mean = closed / word_count,
         speech_rate_mean = word_count / duration) %>%
  group_by(isforrest) %>%
  filter(!is.na(isforrest)) %>%
  summarise(sentiment = mean(sent_mean),
            `syntactic nodes` = mean(comp_mean, na.rm = T),
            `syntactic depth` = mean(depth, na.rm = T),
            `open words` = mean(open_mean),
            `closed words` = mean(closed_mean),
            seconds = mean(duration, na.rm = T),
            `lexical frequency` = mean(lexfreq_norm_log_mean, na.rm = T),
            words = mean(word_count),
            `lexical minimum` = mean(lexmin, na.rm = T),
            `lexical maximum` = mean(lexmax, na.rm = T),
            `lexical diversity` = mean(CTTR, na.rm = T),
            `lowercase characters` = mean(n_lowers, na.rm = T),
            `uppercase characters` = mean(n_caps, na.rm = T),
            `unique words` = mean(n_uq_words, na.rm = T),
            `unique characters` = mean(n_uq_chars, na.rm = T),
            `characters per word` = mean(n_charsperword, na.rm = T),
            `speech rate` = mean(speech_rate_mean, na.rm = T)) %>%
  reshape2::melt() %>%
  ggplot(aes(x = isforrest, y = value, color = isforrest, fill = isforrest)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~variable, scales = "free", ncol = 4) +
  labs(title = "Overview of predictors",
       subtitle = "Mean text predictors per sentence",
       caption = "Data: Master Thesis") +
  ylab("") +
  labs(x = NULL) +
  scale_fill_npg(alpha = 0.7) +
  scale_color_npg() +
  theme_ft_rc() +
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

#### 02 Add and select predictors ####
forrest_clean <- forrest %>%
  filter(duration < max(forrest[forrest$duration > 10,c("duration","isforrest","text")] %>% filter(isforrest == "FORREST") %>% select(duration)))

forrest_clean <- forrest %>%
  mutate(sentiment = sent_sum / word_count,
         nodes = comp_score / word_count,
         open_words = open / word_count,
         closed_words = closed / word_count,
         speech_rate = word_count / duration)

forrest_clean <- forrest_clean %>%
  select(c(CTTR:speech_rate, duration, word_count, lexfreq_norm_log_mean:lexmax,isforrest)) %>%
  filter(!is.infinite(speech_rate)) %>%
  na.omit()

#### 03 set training parameters ####
set.seed(123)
forrest_split <- initial_split(forrest_clean, strata = isforrest)
forrest_train <- training(forrest_split)
forrest_test <- testing(forrest_split)

set.seed(123)
forrest_folds <- vfold_cv(forrest_train, strata = isforrest)

#### 04 define the recipe ####
forrest_recipe <-
  # which consists of the formula (outcome ~ predictors)
  recipe(isforrest ~ ., data = forrest_clean) %>%
  step_naomit(all_predictors()) %>%
  step_zv(all_predictors())

rf_model <-
  # specify that the model is a random forest
  rand_forest() %>%
  # specify parameters that we want to tune
  set_args(mtry = tune(), trees = tune(), min_n = tune()) %>%
  # select the engine that underlies the model
  set_engine("ranger", importance = "impurity") %>%
  # choose either the continuous regression or binary classification mode
  set_mode("classification")
rf_model

#### 05 set the workflow ####
rf_workflow <- workflow() %>%
  # add the recipe
  add_recipe(forrest_recipe) %>%
  # add the model
  add_model(rf_model)
rf_workflow

#### 06 tune hyperparameters ####
doParallel::registerDoParallel()

set.seed(234)
tune_res <- tune_grid(
  rf_workflow,
  resamples = forrest_folds,
  grid = 20
)

save(tune_res, file = "/home/sebastian/Programme/R/Projekte/Sentiment_Analysis/Forrest/tune_res.Rdata")
load("tune_res.Rdata")

tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, mtry, trees) %>%
  pivot_longer(min_n:trees,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  geom_smooth(aes(color = parameter, fill = parameter), se = F, show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC") +
  labs(title = "Overview of hyperparameters",
       subtitle = "It looks like lower values of mtry (number of predictors sampled for splitting at each node) \nand higher values of trees perform well.",
       caption = "Data: Master Thesis") +
  #ylab("") +
  labs(x = NULL) +
  theme_minimal() +
  scale_fill_npg(alpha = 0.7) +
  scale_color_npg() +
  theme_ft_rc() +
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

show_best(tune_res, "roc_auc")

set.seed(123)
rf_grid <- expand.grid(mtry = c(2,3,4), trees = c(1300, 1500, 2000), min_n = c(2,4,6,8,10))
# extract results
rf_tune_results <- rf_workflow %>%
  tune_grid(resamples = forrest_folds,
            grid = rf_grid,
            metrics = metric_set(accuracy, roc_auc, sens, spec) # choose some metrics
  )

save(rf_tune_results, file = "/home/sebastian/Programme/R/Projekte/Sentiment_Analysis/Forrest/rf_tune_results.Rdata")
# load data
load("rf_tune_results.Rdata")

# print results
rf_tune_results %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(trees_fct = as.factor(trees)) %>%
  ggplot(aes(x = mtry, y = mean, fill = trees_fct, color = trees_fct, group = trees_fct)) +
  geom_point() +
  geom_line() +
  labs(title = "Random Forest",
       subtitle = "ROC of random forest models with varying number of trees and split-nodes and minimum terminal nodes",
       caption = "Data: Master Thesis") +
  theme_minimal() +
  ylab("ROC") +
  theme_ft_rc() +
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  scale_fill_npg(alpha = 0.7, name = "trees") +
  scale_color_npg(name = "trees") +
  facet_wrap(~min_n)

#### 07 Finalize the workflow ####
# extract the best value for the roc metric with select_best
param_final <- rf_tune_results %>% select_best(metric = "roc_auc")
rf_workflow <- rf_workflow %>% finalize_workflow(param_final)

#### 08 Evaluate the model on the test set ####
# fit on the training set and evaluate on test set
rf_fit <- rf_workflow %>% last_fit(forrest_split)
# check performance on test set
test_performance <- rf_fit %>% collect_metrics()
# generate predictions from the test set
test_predictions <- rf_fit %>% collect_predictions()
# create confusion matrix
test_predictions %>% conf_mat(truth = isforrest, estimate = .pred_class)
# visualize data
confusion_matrix <- test_predictions %>% conf_mat(truth = isforrest, estimate = .pred_class)
confusion_matrix <- as.data.frame(confusion_matrix["table"])

plotTable <- confusion_matrix %>%
  mutate(goodbad = ifelse(confusion_matrix$table.Prediction == confusion_matrix$table.Truth, "good", "bad")) %>%
  group_by(table.Truth) %>%
  mutate(prop = table.Freq/sum(table.Freq))

ggplot(data = plotTable, mapping = aes(x = table.Truth, y = table.Prediction, fill = goodbad, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = table.Freq), vjust = .5, fontface  = "bold", alpha = 1, color = "white") +
  theme_minimal() +
  ylim(rev(levels(confusion_matrix$table.Prediction))) +
  theme_ft_rc() +
  labs(title = "Confusion matrix",
       subtitle = "How well can we predict the speakers of spoken lines?",
       caption = "Data: Master Thesis") +
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  scale_fill_npg(alpha = 1) +
  scale_color_npg() +
  coord_equal()

test_predictions %>%
  ggplot() +
  geom_density(aes(x = .pred_FORREST, fill = isforrest, color = isforrest), alpha = 0.5) +
  labs(title = "Classification",
       subtitle = "Performance of the selected random forest model",
       caption = "Data: Master Thesis") +
  theme_minimal() +
  theme_ft_rc() +
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  scale_fill_npg(alpha = 0.7) +
  scale_color_npg()

test_predictions %>%
  group_by(id) %>%
  roc_curve(isforrest, .pred_FORREST) %>%
  ggplot(aes(1 - specificity, sensitivity, color = id)) +
  geom_abline(lty = 2, color = "gray80", size = 1.2) +
  geom_path(show.legend = FALSE, alpha = 0.6, size = 1.2) +
    labs(title = "ROC curve",
         subtitle = "Performance of the selected random forest model",
         caption = "Data: Master Thesis") +
    theme_ft_rc() +
    theme(panel.border = element_blank(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank()) +
    scale_fill_npg() +
    scale_color_npg() +
  coord_equal()

#### 09 Fitting the final model and extract important features ####

library(vip)
imp_df <- rf_workflow %>%
  fit(data = forrest_test) %>%
  pull_workflow_fit() %>%
  vi()

imp_df$Variable <- c("lexical frequency",
                     "lexical maximum",
                     "lexical minimum",
                     "speech rate",
                     "seconds",
                     "characters per word",
                     "lowercase characters",
                     "open words",
                     "unique_characters",
                     "closed words",
                     "syntactic nodes",
                     "sentiment",
                     "lexical diversity",
                     "uppercase characters",
                     "unique words",
                     "words",
                     "syntactic depth")

imp_df$Variable <- as.factor(imp_df$Variable)

imp_df %>%
  mutate(Variable = fct_reorder(Variable, Importance)) %>%
  ggplot(aes(x = Variable, y = Importance, color = Importance, fill = Importance)) +
  geom_segment(aes(xend = Variable, yend=0),  show.legend = FALSE) +
  geom_point(size = 4,  show.legend = FALSE) +
  theme_bw() +
  xlab("") +
  coord_flip() +
  labs(title = "Importance of predictors",
       subtitle = "These are the predictors that are most important globally for \nwhether a line was spoken by Forrest or not.",
       caption = "Data: Master Thesis") +
  theme_ft_rc() +
  theme(panel.border = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) +
  scale_fill_gradient(low="#e64b35", high= "#4dbbd5", guide = guide_legend(reverse = TRUE)) +
  scale_color_gradient(low="#e64b35", high="#4dbbd5", guide = guide_legend(reverse = TRUE))
