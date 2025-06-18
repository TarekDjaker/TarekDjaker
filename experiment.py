import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def true_function(x: np.ndarray) -> np.ndarray:
    """Target function generating the data."""
    return np.sin(2 * np.pi * x)


def generate_data(n: int, noise_std: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic one-dimensional data."""
    X = np.random.rand(n, 1)
    y = true_function(X[:, 0]) + noise_std * np.random.randn(n)
    return X, y


def random_features_ls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    K: int,
    B: float,
) -> np.ndarray:
    """Least-squares regression on randomly generated features."""
    beta = np.random.choice([-B, B], size=(1, K))
    gamma = np.random.uniform(-np.pi, np.pi, size=K)

    def phi(X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(X @ beta + gamma)))

    Phi_train = phi(X_train)
    Phi_test = phi(X_test)

    lr = LinearRegression(fit_intercept=True)
    lr.fit(Phi_train, y_train)
    return lr.predict(Phi_test)


def shallow_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    K: int,
    epochs: int = 200,
    lr: float = 0.1,
) -> np.ndarray:
    """Train a one hidden-layer neural network with sigmoid activation."""
    model = Sequential(
        [Dense(K, activation="sigmoid", input_shape=(1,)), Dense(1, activation="linear")]
    )
    model.compile(optimizer=SGD(learning_rate=lr), loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    return model.predict(X_test).flatten()


def cv_choose_K(
    X: np.ndarray, y: np.ndarray, K_list: list[int], B: float
) -> int:
    """Select the best feature dimension using simple cross-validation."""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.5, random_state=1)
    best_K = K_list[0]
    best_err = np.inf
    for K in K_list:
        y_pred = random_features_ls(X_tr, y_tr, X_val, K, B)
        err = np.mean((y_val - y_pred) ** 2)
        if err < best_err:
            best_err = err
            best_K = K
    return best_K


def run_experiment(n_list: list[int], reps: int = 10) -> dict[str, list[float]]:
    """Run the numerical experiment for different sample sizes."""
    results: dict[str, list[float]] = {"nn": [], "rf": [], "cv": []}
    for n in n_list:
        err_nn, err_rf, err_cv = [], [], []
        B = (np.log(n) ** 2) * np.sqrt(n)
        K0 = int(np.sqrt(n))
        K_list = [max(1, int(np.sqrt(n) / k)) for k in [4, 2, 1, 0.5, 0.25]]
        for _ in range(reps):
            X, y = generate_data(n)
            X_test, y_test = generate_data(2000)
            y_pred_nn = shallow_nn(X, y, X_test, K0)
            err_nn.append(np.mean((y_test - y_pred_nn) ** 2))
            y_pred_rf = random_features_ls(X, y, X_test, K0, B)
            err_rf.append(np.mean((y_test - y_pred_rf) ** 2))
            K_star = cv_choose_K(X, y, K_list, B)
            y_pred_cv = random_features_ls(X, y, X_test, K_star, B)
            err_cv.append(np.mean((y_test - y_pred_cv) ** 2))
        results["nn"].append(np.median(err_nn))
        results["rf"].append(np.median(err_rf))
        results["cv"].append(np.median(err_cv))
    return results


def plot_results(n_list: list[int], results: dict[str, list[float]]) -> None:
    """Plot the median MSE in log-log scale."""
    styles = {"nn": "-o", "rf": "-s", "cv": "-^"}
    plt.figure(figsize=(6, 4))
    for key, style in styles.items():
        plt.plot(n_list, results[key], style, label=key)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n (log scale)")
    plt.ylabel("MSE (log scale)")
    plt.legend()
    plt.title(r"Taux $\sim n^{-1/2}$ pour NN, RF et CV")
    plt.grid(which="both")
    plt.show()


def main() -> None:
    np.random.seed(0)
    tf.random.set_seed(0)
    n_list = [200, 500, 1000, 2000, 5000]
    results = run_experiment(n_list)
    plot_results(n_list, results)


if __name__ == "__main__":
    main()
