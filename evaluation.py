import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns

# NON MESSO
def predicted_actual(y_test, y_predicted, model):
    fig, ax = plt.subplots()
    ax.scatter(y_predicted, y_test, color="#1f78b4")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=2)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.title("Predicted vs Actual values for model: " + str(model))
    plt.show()


def residual_plots(y_test, y_predicted, model):
    # Plotting the residuals of y and pred_y
    plt.plot(y_predicted, y_test - y_predicted, 'o', color="#1f78b4")
    plt.hlines(0, xmin=min(y_predicted), xmax=max(y_predicted), color="red")
    plt.show()


def evaluate(y_test, y_predicted, models):
    for i in range(len(y_predicted)):
        print("Metrics to evaluate regression models: ")
        print(models[i])
        print('R^2 is ', r2_score(y_test, y_predicted[i]))
        print('Root Mean Square Error is ', math.sqrt(mean_squared_error(y_test, y_predicted[i])))
        print('Mean Absolute error is ', mean_absolute_error(y_test, y_predicted[i]))
        print("\n")

        d = ['#a6cee3', "#1f78b4"]
        sns.set_palette(sns.color_palette(d))

        residual_plots(y_test, y_predicted[i], models[i])
        #predicted_actual(y_test, y_predicted[i], models[i])
