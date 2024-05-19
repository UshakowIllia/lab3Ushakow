import numpy as np


# Функція активації
def step_function(x):
    return np.where(x >= 0, 1, 0)


# Вхідні дані для XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Встановлюємо ваги та зміщення для прихованого шару
weights_hidden = np.array([[1, -1], [-1, 1]])
bias_hidden = np.array([-0.5, -0.5])

# Встановлюємо ваги та зміщення для вихідного шару
weights_output = np.array([1, 1])
bias_output = -0.5


# Обчислення значень нейронів у прихованому шарі та вихідному шарі
def forward_pass(x):
    hidden_input = np.dot(x, weights_hidden) + bias_hidden
    hidden_output = step_function(hidden_input)
    final_input = np.dot(hidden_output, weights_output) + bias_output
    final_output = step_function(final_input)
    return final_output

# Візуалізація рівнянь розділяючих прямих
def plot_decision_boundary():
    import matplotlib.pyplot as plt

    x = np.linspace(-0.5, 1.5, 400)
    y1 = (0.5 - 1 * x) / -1
    y2 = (0.5 - (-1) * x) / 1

    plt.plot(x, y1, label='Нейрон 1: x1 - x2 = 0.5')
    plt.plot(x, y2, label='Нейрон 2: -x1 + x2 = 0.5')

    plt.scatter(inputs[:, 0], inputs[:, 1], c=[forward_pass(x) for x in inputs], cmap='bwr', edgecolor='k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Двошаровий персептрон для функції XOR')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_decision_boundary()
