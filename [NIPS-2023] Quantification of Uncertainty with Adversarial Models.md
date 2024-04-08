# Quantification of Uncertainty with Adversarial Models

## 1. Problem Definition
Prevalent real-world adoption of deep learning models has increased the demand for the ability to assess the reliability of the predictions of these models, especially in high stake applications. This ability could be considered by quantifying the predictive uncertainty of deep neural networks [1, 2]. Predictive uncertainty can be categorized into two types:
- *Aleatoric*, variability caused by inherent stochasticity of sampling outcomes from the predictive distribution
- *Epistemic*, uncertainty caused by the lack of knowledge of the true model or parameter uncertainty

![Fig 1](https://drive.google.com/uc?id=1BzFEKq4GJLSVHoA1P5r4rtkLMCmKUOGg")

While aleatoric uncertainty cannot be reduced, epistemic uncertainty can be reduced by more data or better models. Thus, knowing how to quantify and evaluate epistemic uncertainty is a crucial element in improving the performance of deep learning models. However, current uncertainty quantification methods like Monte-Carlo (MC) dropout [3] and Deep Ensembles [4] were found to underperform when estimating epistemic uncertainty. The reason for the underperformances was mainly because these methods primarily consider only the posterior counterpart of the integrand defining epistemic uncertainty (see Fig. 1) and are subject to missing the important posterior modes when the whole integrand is large, including when the divergence counterpart is also large. 

## 2. Motivation
This paper discusses two different settings of predictive uncertainty quantification:
* **Expected uncertainty when selecting a model**
  
  The total uncertainty in this setting can be computed as the posterior expectation of the cross-entropy (CE) between the predictive distribution of candidate models, p(y|x, w*), and the Bayesian model average (BMA), p(y|x, D), which can be derived further into the equation shown in Fig 1. Aleatoric uncertainty here represents the expected stochasticity (entropy) of sampling outcomes from the predictive distribution of candidate models p(y|x, w*), while epistemic uncertainty is the mismatch (KL-divergence) between the predictive distributions of candidate models and the BMA.


* **Uncertainty of a given, pre-selected model**
    
  The total uncertainty in this setting can be computed similarly to the former one, with the difference being that the CE between the predictive distribution of a given, pre-selected model, p(y|x, w), and some candidate models, p(y|x, w*), is now computed (see Fig 2).
  
![Fig 2](https://drive.google.com/uc?id=1Ga__00sKf2tJtp0p7IvlruSBZrMHuJQ0)

As shown in Fig 1 and Fig 2, quantifying epistemic uncertainty requires an estimation of the posterior integrals, which are generally approximated using Monte Carlo integration. A fair approximation of these integrals requires not only to capture large values of the posterior but also large values of the KL-divergence. Variational inference [3] and ensembles [4] estimate the posterior integral based on models with high posterior.

Furthermore, all gradient descent-based methods are prone to missing the important posterior modes because they are invariant to the same input attributes. Since gradient descent always starts with attributes with a higher correlation to the target, posterior modes that are located far away from these input attributes’ solution space are almost never found. Other works, such as Markov Chain Monte Carlo sampling approximated by stochastic gradient variants, also face limitations in real-world high-stakes applications in terms of efficiency and escaping local posterior modes.

This paper further aims to contribute to these aspects:
* Introducing a framework to approximate the integral that defines epistemic uncertainty with substantially lower approximation error of the integral estimator than previous methods
* Introducing adversarial models that will have considerably different predictions than a reference model while having similarly high posteriors
* Introducing a new setting of uncertainty quantification by quantifying the predictive uncertainty of a given, pre-selected model


## 3. Proposed Method
Lorem Ipsum Dolor Sit Amet
## 4. Experiment
Lorem Ipsum Dolor Sit Amet
### 4.1 Experiment Setup
Lorem Ipsum Dolor Sit Amet
### 4.2 Experiment Result
Lorem Ipsum Dolor Sit Amet
## 5. Conclusion
Lorem Ipsum Dolor Sit Amet
## Author Information
Dimas Ahmad (dimasat@kaist.ac.kr), Graduate School of Data Science, KAIST
## References
* [Paper link](https://arxiv.org/abs/2307.03217)
* References:
    * [1] Kajetan Schweighofer, Lukas Aichberger, Mykyta Ielanskyi, Gunter Klam-bauer, and Sepp Hochreiter. Quantification of   uncertainty with adversarial models. Advances in Neural Information Processing Systems, 36, 2024.


```python

```
