---
output: 
  html_document:
    keep_md: true
---


```r
#import the packages
library(kknn)
```

```
## Warning: package 'kknn' was built under R version 3.6.2
```

```r
library(tidyverse)
```

```
## -- Attaching packages ----------------------------------------------------------------------------------------------------------------------------------------- tidyverse 1.3.0 --
```

```
## v ggplot2 3.2.1     v purrr   0.3.3
## v tibble  2.1.3     v dplyr   0.8.3
## v tidyr   1.0.0     v stringr 1.4.0
## v readr   1.3.1     v forcats 0.4.0
```

```
## -- Conflicts -------------------------------------------------------------------------------------------------------------------------------------------- tidyverse_conflicts() --
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
```

```r
library(readxl)
library(caret)
```

```
## Loading required package: lattice
```

```
## 
## Attaching package: 'caret'
```

```
## The following object is masked from 'package:purrr':
## 
##     lift
```

```
## The following object is masked from 'package:kknn':
## 
##     contr.dummy
```

```r
#load the data from working directory
Heart = read_excel(path = "/Heart Disease.xlsx")


## Classification with *knn*

# Recode the HeartDisease as a factor
Heart$HeartDisease = factor(Heart$HeartDisease,
                            levels = c("Yes","No"),
                            labels = c("Yes","No"))
# Let's try k = 8
KNNHeart = kknn(HeartDisease ~ ChestPain + Age, train = Heart, test = Heart, k = 8)

# Add predicted values to our dataset
Heart = Heart %>% mutate(HeartDiseaseKNN = KNNHeart$fitted.values)

# Let's take a look at the result
Heart %>% select(HeartDisease, HeartDiseaseKNN)
```

```
## # A tibble: 297 x 2
##    HeartDisease HeartDiseaseKNN
##    <fct>        <fct>          
##  1 No           No             
##  2 Yes          Yes            
##  3 Yes          Yes            
##  4 No           No             
##  5 No           No             
##  6 No           No             
##  7 Yes          Yes            
##  8 No           No             
##  9 Yes          Yes            
## 10 Yes          No             
## # ... with 287 more rows
```
As the model performs classification, the first step in the analysis is to recode the HeartDisease variable into a factor variable. 


```r
Heart %>% group_by(HeartDisease,HeartDiseaseKNN) %>% summarise(Patients = n())
```

```
## # A tibble: 4 x 3
## # Groups:   HeartDisease [2]
##   HeartDisease HeartDiseaseKNN Patients
##   <fct>        <fct>              <int>
## 1 Yes          Yes                  105
## 2 Yes          No                    32
## 3 No           Yes                   29
## 4 No           No                   131
```
The function above shows how to check the accuracy of the model. The knn model predicted 236 out of 297 observations correctly, since we see 32 patients and 29 patients were labeled differently.

The k observation chosen above was k=8. What if we don't know what values of k to choose? Refer to the code below:

```r
# Optimal k
set.seed(12745) # This sets a start value for generating random numbers

KNNHeartOptimal = train.kknn(HeartDisease ~ ChestPain + Age, 
                             data = Heart,
                             kmax = 20) # max k we are interested in trying

# Now we can view the results with the summary function
summary(KNNHeartOptimal)
```

```
## 
## Call:
## train.kknn(formula = HeartDisease ~ ChestPain + Age, data = Heart,     kmax = 20)
## 
## Type of response variable: nominal
## Minimal misclassification: 0.2525253
## Best kernel: optimal
## Best k: 6
```

If we do not know what value of k to choose, the train.kknn function will choose an optimal value of k for us using cross validation.
The output suggests that k=6 would be the best model for our data. 
To fit this model, one would follow the same steps as above, replace k=8 with k=6



```r
## Logistic Regression

# The I() function creates a logical vector that is TRUE when HeartDisease is "Yes"
# and FALSE otherwise
Heart = Heart %>% mutate(HeartDisease = I(HeartDisease == "Yes") %>% as.numeric())

# Let's predict HeartDisease with the Age variable
HeartLogit = glm(HeartDisease ~ Age, # same as in lm()
                 data = Heart,
                 family = "binomial") # for logistic, this is always set to "binomial"

summary(HeartLogit)
```

```
## 
## Call:
## glm(formula = HeartDisease ~ Age, family = "binomial", data = Heart)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.6070  -1.0744  -0.8323   1.1701   1.7105  
## 
## Coefficients:
##             Estimate Std. Error z value Pr(>|z|)    
## (Intercept) -3.05122    0.76862  -3.970  7.2e-05 ***
## Age          0.05291    0.01382   3.829 0.000128 ***
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 409.95  on 296  degrees of freedom
## Residual deviance: 394.25  on 295  degrees of freedom
## AIC: 398.25
## 
## Number of Fisher Scoring iterations: 4
```

The first step in performing a logistic regression is to recode your response variable to numeric value equal 1 for the event of interest and 0 otherwise using the mutate function.
In this case, we will be predicting whether a patient will develop heart disease based on Age.
We must recode the HeartDisease variable in the Heart dataset to be equal to 1 if HeartDisease is “Yes” and 0 otherwise.
The glm() function in R fits generalized linear model, which includes logistic linear regression, and family= “binomial” runs a logistic linear regression.



```r
## Logistic Regression with Multiple Predictors

Heart = Heart %>% mutate(HeartDisease = I(HeartDisease == "Yes") %>% as.numeric())

# Let's predict HeartDisease with the ChestPain, Sex, and Age variables
HeartLogitMultiple = glm(HeartDisease ~ ChestPain + Sex + Age,
                         data = Heart,
                         family = "binomial")
```

```
## Warning: glm.fit: algorithm did not converge
```

```r
# Summary of results
summary(HeartLogitMultiple)
```

```
## 
## Call:
## glm(formula = HeartDisease ~ ChestPain + Sex + Age, family = "binomial", 
##     data = Heart)
## 
## Deviance Residuals: 
##        Min          1Q      Median          3Q         Max  
## -2.409e-06  -2.409e-06  -2.409e-06  -2.409e-06  -2.409e-06  
## 
## Coefficients:
##                       Estimate Std. Error z value Pr(>|z|)
## (Intercept)         -2.657e+01  1.416e+05       0        1
## ChestPainnonanginal -5.407e-16  4.980e+04       0        1
## ChestPainnontypical  1.044e-14  6.019e+04       0        1
## ChestPaintypical     2.274e-13  8.019e+04       0        1
## Sex                  2.114e-14  4.499e+04       0        1
## Age                  1.872e-15  2.344e+03       0        1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 0.0000e+00  on 296  degrees of freedom
## Residual deviance: 1.7231e-09  on 291  degrees of freedom
## AIC: 12
## 
## Number of Fisher Scoring iterations: 25
```

The interpretation of the output above, 
if Gender = 1 (Male), then the odds of developing heart disease are greater. To determine
how much greater, we calculate e^(1.70898) = 5.52. 
This means that patients with Gender=1(Male) have an odds of developing heart disease that is 5.52 times greater than the odds of heart disease for patients with Gender=0(Female).

