# ML Algorithms from Scratch

Each notebook implements a classical ML algorithm in NumPy and compares against the scikit-learn version.

## Adaboost.R2 — [`adaboost_r2.ipynb`](adaboost_r2.ipynb)
Boosted regression with weighted stumps. For iteration $m$:
$$L_i = \frac{|y_i - \hat y_i|}{\max_j |y_j - \hat y_j|}, \quad \bar L_m = \sum_i w_i L_i, \quad \beta_m = \frac{\bar L_m}{1 - \bar L_m}, \quad \alpha_m = \ln \tfrac{1}{\beta_m}$$
Sample weights update as $w_i \leftarrow w_i\,\beta_m^{\,1 - L_i}$, and the final prediction is the weighted median of weak learners.

## Decision Tree Regressor (CART) — [`decision_tree_regressor.ipynb`](decision_tree_regressor.ipynb)
At each node pick the split $(j, t)$ that maximizes variance reduction:
$$\Delta = \mathrm{MSE}(S) - \tfrac{|S_L|}{|S|}\mathrm{MSE}(S_L) - \tfrac{|S_R|}{|S|}\mathrm{MSE}(S_R), \quad \mathrm{MSE}(S) = \tfrac{1}{|S|}\sum_{i \in S}(y_i - \bar y_S)^2$$

## K-Means++ — [`kmeans_plus_plus.ipynb`](kmeans_plus_plus.ipynb)
Seed centroid $c_{k+1}$ with probability $\propto D(x)^2$ where $D(x) = \min_{j \le k} \|x - c_j\|$, then alternate
$$\text{assign: } z_i = \arg\min_k \|x_i - c_k\|^2, \quad \text{update: } c_k = \tfrac{1}{|C_k|}\sum_{i \in C_k} x_i.$$

## KNN Regressor — [`knn_regressor.ipynb`](knn_regressor.ipynb)
For a query $x$, let $\mathcal{N}_k(x)$ be its $k$ nearest training points. Uniform / inverse-distance weighted prediction:
$$\hat y(x) = \frac{1}{k}\sum_{i \in \mathcal{N}_k(x)} y_i, \qquad \hat y_w(x) = \frac{\sum_i w_i y_i}{\sum_i w_i},\; w_i = \tfrac{1}{d(x, x_i) + \varepsilon}.$$

## Linear Discriminant Analysis — [`lda.ipynb`](lda.ipynb)
Solve the generalized eigenproblem $S_B w = \lambda S_W w$, where
$$S_W = \sum_c \sum_{i: y_i = c}(x_i - \mu_c)(x_i - \mu_c)^\top, \quad S_B = \sum_c n_c (\mu_c - \mu)(\mu_c - \mu)^\top.$$
Top eigenvectors of $S_W^{-1} S_B$ give the discriminant directions.

## Logistic Regression (softmax) — [`logistic_regression.ipynb`](logistic_regression.ipynb)
$$P(y = k \mid x) = \frac{e^{w_k^\top x}}{\sum_j e^{w_j^\top x}}, \qquad \mathcal{L} = -\tfrac{1}{N}\sum_{i,k} y_{ik} \log P(y = k \mid x_i) + \tfrac{\lambda}{2}\|W\|_F^2.$$
Trained with gradient descent on $\nabla_{w_k} \mathcal{L} = \tfrac{1}{N} X^\top(\hat Y - Y) + \lambda w_k$.

## Multilayer Perceptron — [`multilayer_perceptron.ipynb`](multilayer_perceptron.ipynb)
Forward: $a^{(l)} = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})$. Backprop (MSE loss):
$$\delta^{(L)} = \hat y - y, \quad \delta^{(l)} = (W^{(l+1)})^\top \delta^{(l+1)} \odot \sigma'(z^{(l)}), \quad \nabla_{W^{(l)}} \mathcal{L} = \delta^{(l)} (a^{(l-1)})^\top.$$

## Gaussian Naive Bayes — [`naive_bayes_classifier.ipynb`](naive_bayes_classifier.ipynb)
$$\hat y = \arg\max_c \Big[ \log P(c) + \sum_j \log \mathcal{N}(x_j \mid \mu_{c,j}, \sigma_{c,j}^2) \Big], \quad \mathcal{N}(x \mid \mu, \sigma^2) = \tfrac{1}{\sqrt{2\pi}\sigma} \exp\!\big(-\tfrac{(x - \mu)^2}{2\sigma^2}\big).$$

## RBF Kernel Regression (Nadaraya–Watson) — [`rbf_kernel_regression.ipynb`](rbf_kernel_regression.ipynb)
$$\hat y(x) = \frac{\sum_i K(x, x_i)\, y_i}{\sum_i K(x, x_i)}, \quad K(x, x_i) = \exp\!\Big(-\tfrac{\|x - x_i\|^2}{2\sigma^2}\Big).$$

## Support Vector Regression — [`svm_regressor.ipynb`](svm_regressor.ipynb)
$\varepsilon$-insensitive regression:
$$\min_{w, b, \xi, \xi^*} \tfrac{1}{2}\|w\|^2 + C\sum_i (\xi_i + \xi_i^*) \;\;\text{s.t.}\;\; |y_i - (w^\top \phi(x_i) + b)| \le \varepsilon + \xi_i^{(*)},\; \xi_i^{(*)} \ge 0.$$
Linear and RBF kernels $\phi$ are both swept over $(C, \varepsilon, \gamma)$.
