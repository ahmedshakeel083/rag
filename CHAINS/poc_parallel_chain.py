from langchain_core.runnables import RunnableParallel
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation", 
    pipeline_kwargs={
        "max_length": 4096,
        "temperature": 0.7,
    },
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt_one = PromptTemplate(
    template = ("""
    Create notes about the below report 
    Report: {text}."""),
    input_variables = ["text"]
)

prompt_two = PromptTemplate(
    template = ("""
    Create a question answer style report about {text}."""),
    input_variables = ["text"]
)

prompt_three = PromptTemplate(
    template = ("""
    Merge the provided documents into a single report: {notes} and {quiz}."""),
    input_variables = ["notes", "quiz"]
)

parallel_chain = RunnableParallel(
    {
        "notes": prompt_one | model | parser,
        "quiz": prompt_two | model | parser,
    }
)

merge_chain = prompt_three | model | parser

chain = parallel_chain | merge_chain

text = """
# Linear Regression: A Detailed Report

## 1. Overview

Linear Regression is a foundational supervised learning algorithm that models the relationship between one or more independent variables (predictors) and a continuous dependent variable (target) by fitting a linear equation to observed data. Despite its simplicity, it remains one of the most widely used techniques in statistics and machine learning, serving as both a standalone predictive tool and a building block for more complex models.

---

## 2. Mathematical Formulation

### 2.1 Simple Linear Regression

With one predictor:

$$
y = \beta_0 + \beta_1 x + \varepsilon
$$

- $y$: dependent variable
- $x$: independent variable
- $\beta_0$: intercept
- $\beta_1$: slope (coefficient)
- $\varepsilon$: error term (residual), assumed $\varepsilon \sim \mathcal{N}(0, \sigma^2)$

### 2.2 Multiple Linear Regression

With $p$ predictors:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon
$$

In matrix form:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

where $\mathbf{X}$ is the $n \times (p+1)$ design matrix (including a column of ones for the intercept), $\boldsymbol{\beta}$ is the coefficient vector, and $\mathbf{y}$ is the $n \times 1$ response vector.

---

## 3. Assumptions

Linear regression relies on several classical assumptions (the Gauss–Markov assumptions):

| Assumption | Description |
|---|---|
| **Linearity** | The relationship between predictors and target is linear in parameters |
| **Independence** | Observations are independent of each other |
| **Homoscedasticity** | Constant variance of residuals across all levels of predictors |
| **Normality of errors** | Residuals are normally distributed (needed for inference, not for OLS estimation itself) |
| **No perfect multicollinearity** | Predictors are not perfectly linearly related |
| **Exogeneity** | $E[\varepsilon \mid \mathbf{X}] = 0$ — no correlation between errors and predictors |

Violations of these assumptions affect the validity of inference (p-values, confidence intervals) more than the point estimates themselves.

---

## 4. Estimation Methods

### 4.1 Ordinary Least Squares (OLS)

OLS minimizes the sum of squared residuals:

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2
$$

Closed-form solution (normal equations):

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

**Properties (Gauss–Markov Theorem):** Under the classical assumptions, OLS is the Best Linear Unbiased Estimator (BLUE) — minimum variance among all linear unbiased estimators.

**Computational note:** The normal equation requires inverting $\mathbf{X}^\top\mathbf{X}$, which is $O(p^3)$ and numerically unstable when predictors are collinear. In practice, QR decomposition or SVD is preferred.

### 4.2 Gradient Descent

For large datasets where the closed-form solution is expensive, iterative optimization is used:

$$
\beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} J(\boldsymbol{\beta})
$$

where $J(\boldsymbol{\beta}) = \frac{1}{2n}\sum_{i=1}^n (h_\theta(x_i) - y_i)^2$ and $\alpha$ is the learning rate. Variants include Batch, Stochastic (SGD), and Mini-batch gradient descent.

### 4.3 Maximum Likelihood Estimation (MLE)

Under the assumption of normally distributed errors, MLE yields the same coefficient estimates as OLS. This equivalence connects linear regression to probabilistic modeling.

### 4.4 Regularized Variants

| Method | Objective | Effect |
|---|---|---|
| **Ridge (L2)** | $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_2^2$ | Shrinks coefficients; handles multicollinearity |
| **Lasso (L1)** | $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_1$ | Produces sparse solutions (feature selection) |
| **Elastic Net** | Combines L1 and L2 penalties | Balances shrinkage and sparsity |

Ridge has a closed-form solution: $\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$. Lasso requires iterative solvers (coordinate descent, LARS).

---

## 5. Model Evaluation

### 5.1 Goodness of Fit

- **R² (coefficient of determination):** proportion of variance in $y$ explained by the model.
  $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$
- **Adjusted R²:** penalizes for number of predictors; useful for comparing models with different feature counts.
- **Residual Standard Error (RSE):** estimate of $\sigma$.

### 5.2 Error Metrics

- **MSE** = $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$
- **RMSE** = $\sqrt{\text{MSE}}$
- **MAE** = $\frac{1}{n}\sum |y_i - \hat{y}_i|$

### 5.3 Diagnostic Plots

- **Residuals vs. fitted values** — check linearity and homoscedasticity
- **Q–Q plot** — check normality of residuals
- **Scale–location plot** — check variance stability
- **Leverage / Cook's distance** — identify influential points
- **VIF (Variance Inflation Factor)** — detect multicollinearity (VIF > 10 typically flagged)

### 5.4 Statistical Inference

- **t-tests** for individual coefficients: $H_0: \beta_j = 0$
- **F-test** for overall model significance
- **Confidence intervals** for coefficients and predictions
- **Prediction intervals** (wider than confidence intervals; account for individual observation noise)

---

## 6. Practical Workflow

1. **Exploratory data analysis** — scatter plots, correlation matrix, distribution checks
2. **Feature engineering** — transformations (log, polynomial), interaction terms, encoding categoricals (dummy/one-hot)
3. **Train/test split** or cross-validation
4. **Fit the model** (OLS or regularized)
5. **Diagnostics** — residual analysis, VIF, influence measures
6. **Refinement** — remove/transform features, address outliers, consider regularization
7. **Evaluate** on held-out data
8. **Interpret** coefficients (with caution: correlation ≠ causation)

---

## 7. Strengths and Limitations

### Strengths
- **Interpretability:** coefficients have direct, quantifiable meaning
- **Computational efficiency:** fast to train and predict, even on large data
- **Statistical foundation:** well-understood inference, confidence intervals
- **Baseline:** strong benchmark for comparing more complex models
- **Extensibility:** basis for GLMs, regularized models, and neural network layers

### Limitations
- **Linearity assumption:** cannot capture non-linear relationships without manual feature engineering
- **Sensitivity to outliers:** OLS is sensitive due to squared loss
- **Multicollinearity:** unstable coefficient estimates when predictors are correlated
- **Homoscedasticity requirement:** violates inference validity when variance is non-constant (remedies: weighted least squares, robust standard errors)
- **Extrapolation risk:** predictions outside the training range are unreliable
- **No built-in feature selection** (except Lasso)

---

## 8. Extensions and Related Models

- **Polynomial regression:** add polynomial terms of predictors
- **Generalized Linear Models (GLMs):** logistic (binary classification), Poisson (counts), etc.
- **Robust regression:** Huber loss, RANSAC — reduce outlier sensitivity
- **Quantile regression:** model conditional quantiles rather than the mean
- **Bayesian linear regression:** place priors on coefficients; yields posterior distributions (Ridge is equivalent to a Gaussian prior)
- **Mixed-effects / hierarchical models:** for grouped or panel data
- **Partial Least Squares (PLS):** for high-dimensional, collinear predictors
- **Kernel methods:** kernel ridge regression for non-linear relationships without explicit feature mapping

---

## 9. Implementation Notes

- **scikit-learn:** `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` — clean API, cross-validation utilities
- **statsmodels:** `OLS` — full statistical summary (p-values, CIs, diagnostics)
- **Numerical stability:** prefer `lstsq`/QR over explicit matrix inversion; standardize features before regularization
- **Categorical variables:** use one-hot encoding with `drop_first=True` to avoid dummy variable trap
- **Feature scaling:** not required for OLS but essential for regularized variants and gradient descent

---

## 10. Summary

Linear regression is a mature, interpretable, and computationally efficient method for modeling continuous outcomes. Its closed-form solution, statistical inference framework, and well-understood assumptions make it a default first choice for regression problems and a benchmark against which more complex models are evaluated. Its main limitations — linearity, outlier sensitivity, and multicollinearity — are addressable through feature engineering, regularization, and robust estimation techniques, which extend the core framework into a rich family of models used across statistics, econometrics, and machine learning.
"""

output = chain.invoke({"text": text})

print(output)