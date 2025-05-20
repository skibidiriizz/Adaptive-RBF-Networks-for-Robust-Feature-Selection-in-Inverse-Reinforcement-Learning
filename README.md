# Adaptive RBF Networks for Robust Feature Selection in Inverse Reinforcement Learning

This repository contains the code and resources for our project on applying **adaptive Radial Basis Function (RBF) networks** to improve feature selection and reward approximation in **Inverse Reinforcement Learning (IRL)**.

## 🧠 Overview

We propose a hybrid IRL framework that extends polynomial feature-based approaches by introducing:

- **Adaptive RBF Networks** for capturing complex, non-linear reward structures.
- **Multi-Armed Bandit (MAB)**-based center selection for efficient and automated feature relevance.
- **Four Kernel Width Adaptation Techniques** to prevent overfitting and better represent the reward function.

Our approach shows significant improvement in environments like `Pendulum-v1` and `CartPole-v1` over traditional polynomial methods.


## ⚙️ Methodology Highlights

- **Reward Representation:**
  \[
  R(s) = \sum_{i=1}^{K} w_i \cdot \exp\left( -\frac{||s - c_i||^2}{2\sigma_i^2} \right)
  \]

- **Kernel Width Methods:**
  - Density-Adaptive
  - Cluster-Adaptive
  - Covariance-Aware
  - Learned (gradient-based)

- **Center Selection:** 
  - Uses UCB (Upper Confidence Bound) bandit strategy to choose the optimal number of RBF centers.

## 📊 Results

| Environment   | Best K | Silhouette Score | Final Loss |
|---------------|--------|------------------|------------|
| Pendulum-v1   | 3      | 0.8534           | 0.2744     |
| CartPole-v1   | 2      | 0.3552           | 0.1022     |

Our method achieved:
- Robust generalization in non-linear environments.
- Efficient reward recovery even from limited or noisy expert data.



## 👨‍💻 Contributors

- **Ashiq (CS22B2021)**: RBF network design and feature engineering
- **Ajmal (CS22B2046)**: Multi-Armed Bandit for center selection
- **Abishek (CS22B2054)**: Kernel width adaptation strategies

## 📄 License

This project is for academic/research purposes only.

## 📚 References

1. Ng & Russell, ICML 2000
2. Abbeel & Ng, ICML 2004
3. Baimukashev et al., *Automated Feature Selection for IRL*, [arXiv:2403.15079](https://arxiv.org/abs/2403.15079)

---

