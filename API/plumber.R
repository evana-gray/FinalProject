#
# This is a Plumber API. You can run the API by clicking
# the 'Run API' button above.
#
# Find out more about building APIs with Plumber here:
#
#    https://www.rplumber.io/
#

library(plumber)
library(tidyverse)
library(tidymodels)
library(rsample)
library(parsnip)
library(tree)
library(rpart)
library(rpart.plot)
library(baguette)
library(ranger)
library(yardstick)
library(future)

set.seed(16)


#* @apiTitle Evan Gray - Final Project API
#* @apiDescription description



#Read Data

diabetes <- read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

#convert col names to lower case for easy reference
colnames(diabetes) <- tolower(colnames(diabetes))

#Create Factors
diabetes <- diabetes |>
  mutate(
    diabetes_binary_f = factor(diabetes_binary, levels = c(0,1),
                               labels = c("no diabetes", "diabetes")), 
    highbp_f = factor(highbp, levels = c(0,1),
                      labels = c("no high BP", "high BP")),
    highchol_f = factor(highchol, levels = c(0,1),
                        labels = c("no high cholesterol", "high cholesterol")),
    cholcheck_f = factor(cholcheck, levels = c(0,1),
                         labels = c("no cholesterol check in 5 years", "yes cholesterol check in 5 years")),
    smoker_f = factor(smoker, levels = c(0,1),
                      labels = c("no", "yes")), 
    stroke_f = factor(stroke, levels = c(0,1),
                      labels = c("no", "yes")),
    heartdiseaseorattack_f = factor(heartdiseaseorattack, levels = c(0,1),
                                    labels = c("no", "yes")),
    physactivity_f = factor(physactivity, levels = c(0,1),
                            labels = c("no", "yes")),
    fruits_f = factor(fruits, levels = c(0,1),
                      labels = c("no","yes")),
    veggies_f = factor(veggies, levels = c(0,1),
                       labels = c("no", "yes")),
    hvyalcoholconsump_f = factor(hvyalcoholconsump, levels = c(0,1),
                                 labels = c("no", "yes")),
    anyhealthcare_f = factor(anyhealthcare, levels = c(0,1),
                             labels = c("no", "yes")),
    nodocbccost_f = factor(nodocbccost, levels = c(0,1),
                           labels = c("no","yes")),
    genhlth_f = factor(genhlth, levels = 1:5,
                       labels = c("excellent", "very good", "good", "fair", "poor")),
    diffwalk_f  = factor(diffwalk, levels = c(0,1),
                         labels = c("no", "yes")),
    sex_f = factor(sex, levels = c(0,1),
                   labels = c("female", "male")),
    age_f = factor(age, levels = 1:13,
                   labels = c("18 to 24",
                              "25 to 29", 
                              "30 to 34", 
                              "35 to 39",
                              "40 to 44",
                              "45 to 49",
                              "50 to 54",
                              "55 to 59",
                              "60 to 64",
                              "65 to 69",
                              "70 to 74",
                              "75 to 79", 
                              "80 to high"
                   )),
    education_f = factor(education, levels = 1:6,
                         labels = c("Never attended school or only kindergarten",
                                    "Grades 1 through 8 (Elementary)",
                                    "Grades 9 through 11 (Some high school)",
                                    "Grade 12 or GED (High school graduate)",
                                    "College 1 year to 3 years (Some college or technical school)",
                                    "College 4 years or more (College graduate)")
    ),
    income_f = factor(income, levels = 1:8,
                      labels = c("< 10k",
                                 "< 15k",
                                 "< 20k",
                                 "< 25k",
                                 "< 35k",
                                 "< 50k",
                                 "< 75k",
                                 "75k+"))
  )


#Subset Variables of Interest
diabetes_model_data <- diabetes |>
  select(diabetes_binary_f, highbp_f, sex_f, income_f, age_f, bmi)
rm(diabetes) #remove large original file

#split data and fold
diabetes_split <- rsample::initial_split(diabetes_model_data, prop = .7)
diabetes_train <- rsample::training(diabetes_split)
diabetes_test <- rsample::testing(diabetes_split)
diabetes_cv_folds <- vfold_cv(diabetes_train, 5)

reci_1 <- recipe(diabetes_binary_f~ ., data = diabetes_train) |>
  step_normalize(all_numeric(), -diabetes_binary_f) |>
  update_role(diabetes_binary_f, new_role = "outcome") |>
  step_dummy(highbp_f, sex_f, income_f, age_f)

rf3_diabetes <- rand_forest(mtry = 5, trees = 100) |>
  set_engine("ranger") |>
  set_mode("classification")

rf3_flow <- workflow() |>
  add_recipe(reci_1) |>
  add_model(rf3_diabetes)

rf3_fit <- rf3_flow |>
  tune_grid(resamples = diabetes_cv_folds,
            metrics = metric_set(mn_log_loss)
  )

rf3_lowest_log <- select_best(rf3_fit)

rf3_flow2 <- rf3_flow |> 
  finalize_workflow(rf3_lowest_log)

rf3_final_fit <- rf3_flow2 |>
  last_fit(diabetes_split, metrics = metric_set(accuracy, mn_log_loss))

rf_lastfit <- extract_workflow(rf3_final_fit)

#* Make a Prediction
#* @param sex_value
#* @param income_value
#* @param age_value
#* @param highbp_value
#* @param bmi_value
#* @get /predict
function(sex_value = "female", income_value = "75k+", age_value = "60 to 64", highbp_value = "no high BP", bmi_value = 28){
  
  #Per email/instructions - create tibble with same structure as original, but with avg / most common values
  #copy tibble, keep no rows but all columns, and repopulate original tibble to ensure structure is the same
  diabetes_model_avg <- diabetes_model_data[0,] |> 
    add_row(
      highbp_f = highbp_value, 
      sex_f = sex_value, 
      income_f = income_value, 
      age_f = age_value, 
      bmi = as.numeric(bmi_value)
    )
  
  prediction <- predict(rf_lastfit, new_data = diabetes_model_avg)
  print(prediction)
}