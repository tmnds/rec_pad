import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error


def get_metrics_error(test_target, pred_test):

    mse = mean_squared_error(test_target, pred_test)
    rmse = root_mean_squared_error(test_target, pred_test)
    mae = mean_absolute_error(test_target, pred_test)
    mape = mean_absolute_percentage_error(test_target, pred_test)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def get_plot_error(test_target, pred_test):
    erro = test_target - pred_test

    plt.plot(erro)
    plt.title("Erro ao longo do tempo")
    plt.show()


def get_error_distribution(test_target, pred_test):
    erro = test_target - pred_test

    sns.histplot(erro, kde=True)
    plt.title("Distribuição do erro")
    plt.show()

def get_prediction_plot(test_target, pred_test, dataset='Name'):

    plt.figure(figsize=(12,6))
    x = np.arange(len(test_target))

    plt.plot(x, test_target, label='Real', linewidth=2)
    plt.plot(x, pred_test, label='Predito', linestyle='--', linewidth=2)

    plt.title(f"{dataset}", fontsize=14)
    plt.xlabel("Tempo")
    plt.ylabel("Valor")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def get_prediction_plot_full(test_input, test_target, pred_test):
    
    fig, ax = plt.subplots()
    plt.xlabel('Tempo (Diário)')
    plt.ylabel('Temperatura Mínima (°C)')
    x = np.arange(len(test_target))

    plt.plot(test_input, color='black', label='Training Data')
    plt.plot(test_target, color='blue', label='Test Data')
    plt.plot(pred_test, color='red', label='Forecast')
    # plt.legend(bbox_to_anchor=(1.05, 1))
    # ax.xaxis.set_major_formatter(DateFormatter('%Y'))

