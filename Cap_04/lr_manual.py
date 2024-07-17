# %%
import numpy as np

# %%
# Definindo os dados
X = np.array([
    [0.5, 1.0],
    [1.5, 0.5],
    [1.0, 1.5],
    [2.0, 1.0]
])
y = np.array([1, 0, 1, 0])

# %%
# Função Sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# %%
# Função de Custo
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = -1/m * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h)))
    return cost

# %%
# Gradiente Descendente
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = 1/m * X.T.dot(h - y)
        theta = theta - learning_rate * gradient
        if i % 100 == 0:  # Print cost every 100 iterations
            print(f'Cost after iteration {i}: {compute_cost(X, y, theta)}')
    return theta

# %%
# Adicionando uma coluna de 1s para o termo de intercepto
X_b = np.c_[np.ones((X.shape[0], 1)), X]
X_b

# %%
# Inicializando os pesos
theta_initial = np.zeros(X_b.shape[1])
theta_initial

# %%
# Definindo taxa de aprendizado e número de iterações
learning_rate = 0.1
num_iterations = 1000

# %%
# Treinando o modelo
theta_final = gradient_descent(X_b, y, theta_initial, learning_rate, num_iterations)
print('Theta final:', theta_final)

# %%
# Função de Previsão
def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Adicionar termo de intercepto
    return sigmoid(X_b.dot(theta)) >= 0.5

# %%
# Fazendo previsões
predictions = predict(X, theta_final)
print('Previsões:', predictions)

# %%
