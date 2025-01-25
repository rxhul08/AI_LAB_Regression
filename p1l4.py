import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load independent variables (X)
X_values = np.array([
    [3.8915, 4.2105], [3.6898, 6.6051], [2.7763, 7.5057], [3.1137, 5.7724], [2.9251, 5.4315],
    [3.6699, 6.4406], [2.8404, 3.8136], [3.7729, 5.2398], [2.6465, 3.4946], [4.0902, 5.9298],
    [3.3337, 5.5294], [1.44, 5.8302], [3.6919, 5.0708], [4.4506, 3.629], [4.7716, 6.4982],
    [3.7306, 4.8439], [4.9867, 5.6805], [4.1954, 6.455], [5.6164, 6.0755], [3.7672, 4.6705],
    [3.982, 5.2395], [3.9381, 5.2835], [4.0603, 6.4953], [4.3357, 6.7917], [4.5707, 4.4346],
    [2.5098, 4.4806], [2.2003, 5.6314], [4.8419, 5.4988], [4.4708, 5.7022], [2.6502, 4.4475],
    [3.4506, 4.1548], [5.3572, 5.4207], [2.3391, 6.7416], [3.8305, 6.1357], [2.1096, 5.3812],
    [3.674, 5.1154], [3.8091, 4.3737], [3.3172, 6.4038], [4.4469, 6.3588], [3.3633, 5.3338],
    [4.6922, 5.8894], [4.6014, 5.774], [3.2233, 6.4315], [3.7617, 7.1484], [3.7115, 3.7335],
    [4.4714, 5.1916], [4.7502, 5.4191], [4.2977, 4.2184], [3.5279, 5.2086], [2.6096, 6.3162],
    [4.7394, 2.6582], [3.3908, 3.4965], [4.4767, 2.9949], [5.7886, 1.6549], [4.735, 2.4475],
    [6.4513, 1.5372], [4.94, 5.1969], [5.7323, 3.7426], [4.9014, 4.1027], [6.1341, 3.7068],
    [3.8544, 4.6233], [7.1477, 5.0198], [5.2805, 3.0939], [5.3616, 2.5338], [4.0691, 4.6088],
    [5.4984, 5.7533], [5.4295, 3.2801], [4.2903, 2.7389], [6.0986, 3.8345], [7.5195, 2.5559],
    [6.6949, 2.7201], [5.4368, 4.1784], [6.0597, 3.28], [5.715, 4.0286], [5.922, 2.9962],
    [3.4284, 4.0874], [4.5904, 3.8441], [5.2147, 4.0196], [6.1363, 4.6772], [5.8662, 4.3752],
    [5.3976, 1.956], [3.3328, 5.3288], [4.0697, 2.2547], [6.8436, 1.732], [5.638, 4.4592],
    [5.5175, 2.7173], [5.1615, 1.6006], [8.4153, 3.4567], [4.8131, 5.3435], [5.3576, 4.3978],
    [6.0388, 4.195], [6.8282, 2.4295], [5.369, 2.6921], [6.4701, 4.4246], [6.4986, 4.8292],
    [5.7629, 4.6161], [4.1817, 4.0577], [6.3065, 2.4776], [5.6043, 3.3146], [7.0567, 4.7346]
])
X = pd.DataFrame(X_values, columns=["Feature_1", "Feature_2"])

# load dependent variable (y)
file_path = "/Users/rahullenka/Downloads/logisticY.csv"  
data = pd.read_csv(file_path, header=None)
y = data.iloc[:, 0]

# add intercept to X
X["Intercept"] = 1

# normalize features
X_normalized = X.copy()
X_normalized[["Feature_1", "Feature_2"]] = (
    X_normalized[["Feature_1", "Feature_2"]] - X_normalized[["Feature_1", "Feature_2"]].mean()
) / X_normalized[["Feature_1", "Feature_2"]].std()

# align x and y sizes
X_normalized = X_normalized.iloc[:len(y)]

# define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# define cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = (-1 / m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1 - h)))
    return cost

# perform gradient descent
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= alpha * gradient
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

# prepare data for training
X_train = X_normalized.values
y_train = y.values

# initialize parameters and train the model
alpha = 0.1
num_iters = 1000
theta_initial = np.zeros(X_train.shape[1])
theta_optimal, cost_history = gradient_descent(X_train, y_train, theta_initial, alpha, num_iters)

# plot cost vs. iterations
plt.figure(figsize=(8, 6))
plt.plot(range(len(cost_history)), cost_history, label="Learning Rate: 0.1")
plt.title("Cost Function vs. Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.grid()
plt.show()

# plot decision boundary
def plot_decision_boundary(X, y, theta):
    plt.figure(figsize=(8, 6))
    for i in np.unique(y):
        plt.scatter(X[y == i, 0], X[y == i, 1], label=f"Class {int(i)}")

    x_values = [X[:, 0].min(), X[:, 0].max()]
    y_values = -(theta[0] * np.array(x_values) + theta[2]) / theta[1]
    plt.plot(x_values, y_values, label="Decision Boundary", color="red")

    plt.title("Dataset with Decision Boundary")
    plt.xlabel("Feature 1 (Normalized)")
    plt.ylabel("Feature 2 (Normalized)")
    plt.legend()
    plt.grid()
    plt.show()

plot_decision_boundary(X_train[:, :-1], y_train, theta_optimal)

# confusion matrix and metrics
def predict(X, theta, threshold=0.5):
    probabilities = sigmoid(np.dot(X, theta))
    return (probabilities >= threshold).astype(int)

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tp, fn], [fp, tn]])

y_pred = predict(X_train, theta_optimal)
cm = confusion_matrix(y_train, y_pred)
tp, fn = cm[0]
fp, tn = cm[1]

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")
