# Task 7: Support Vector Machines (SVM)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset (Breast Cancer)
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42
)

# 2. Train SVM with Linear Kernel
linear_svm = SVC(kernel='linear', C=1)
linear_svm.fit(X_train, y_train)

# 3. Train SVM with RBF Kernel
rbf_svm = SVC(kernel='rbf', C=1, gamma=0.5)
rbf_svm.fit(X_train, y_train)

# 4. Cross-validation scores
linear_scores = cross_val_score(linear_svm, X_pca, y, cv=5)
rbf_scores = cross_val_score(rbf_svm, X_pca, y, cv=5)

print("Linear SVM CV Accuracy:", np.mean(linear_scores))
print("RBF SVM CV Accuracy:", np.mean(rbf_scores))

# 5. Visualization function (no plt.show() here)
def plot_decision_boundary(clf, X, y, title, filename):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.savefig(filename)  # Save figure
    print(f"Saved plot as {filename}")

# 6. Create and save both plots
plt.figure(figsize=(8, 6))
plot_decision_boundary(linear_svm, X_train, y_train, "Linear SVM Decision Boundary", "linear_svm_plot.png")

plt.figure(figsize=(8, 6))
plot_decision_boundary(rbf_svm, X_train, y_train, "RBF SVM Decision Boundary", "rbf_svm_plot.png")

plt.show()
