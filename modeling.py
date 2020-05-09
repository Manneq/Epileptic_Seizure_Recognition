import pandas as pd
import statsmodels.tsa.arima_model
import numpy as np
import data_management
import plotting

"""
def arima_model_evaluation(training_set, validation_set, arima_order):
    history = \
        [brain_activity for brain_activity in training_set.values.tolist()]
    predictions = []

    for i in range(len(validation_set.values.tolist())):
        arima_model = \
            statsmodels.tsa.arima_model.ARIMA(history,
                                              order=arima_order).fit(disp=0)

        predictions.append(arima_model.forecast()[0])
        history.append(validation_set.values.tolist()[i])

    return sklearn.metrics.mean_squared_error(validation_set.values.tolist(),
                                              predictions)


def arima_model_grid_search(training_set, validation_set):
    best_mse_error, best_arima_order = float("inf"), None

    for p_value in range(1, 3):
        for d_value in range(1, 3):
            for q_value in range(1, 3):
                mse_error = \
                    arima_model_evaluation(training_set, validation_set,
                                           (p_value, d_value, q_value))
                print(mse_error)

                if mse_error < best_mse_error:
                    best_mse_error, best_arima_order = \
                        mse_error, (p_value, d_value, q_value)

    return best_arima_order
"""


def forecasting_using_arima(data):
    """
        Method to forecast patients brain activity using ARIMA model.
        param:
            data - pandas DataFrame of initial data
    """
    # Initial data conversion to time series
    data = data_management.time_series_conversion(data)

    # Time for forecasting
    forecast_time = np.arange(23, 24, 1 / 179)

    # Forcast bbrain activity of first 2 patients
    for patient in data['patients'].unique().tolist()[:2]:
        patient_data = data.loc[data['patients'] == patient].\
            drop(columns=['patients'], axis=1).set_index('time').sort_index()

        """
        training_set, validation_set = \
            patient_data.iloc[:patient_data.shape[0] -
                              round(patient_data.shape[0] * 0.2), :], \
            patient_data.iloc[patient_data.shape[0] -
                              round(patient_data.shape[0] * 0.2):, :]
        """

        # ARIMA model creation and training
        arima_model = \
            statsmodels.tsa.arima_model.ARIMA(patient_data.values.tolist(),
                                              order=(25, 1, 0)).fit(disp=0)

        # Brain activity forecasting
        forecasted_patient_data = \
            pd.DataFrame(arima_model.forecast(steps=forecast_time.shape[0])[0], 
                         columns=['brain_activity'], index=forecast_time)

        # Time series plotting
        plotting.line_plotting(patient_data, forecasted_patient_data,
                               ["Time, s", "Brain activity"],
                               "Brain activity of patient " + str(patient))

    return


"""
def bayesian_network_modeling(data):
    modeling_model = pomegranate.BayesianNetwork.from_samples(
        data.drop(data.iloc[:, 1:175], axis=1), algorithm='chow-liu',
        n_jobs=-1)

    plt.figure(figsize=(1920 / 96, 1080 / 96), dpi=96)
    plt.title('Graph')
    nx.draw_shell(modeling_model.graph, with_labels=True)
    plt.savefig("graph_shell.png")
    plt.close()

    return
    """


def modeling_applying(data):
    """
        Method to apply modeling.
        param:
            data - pandas DataFrame of initial data.
    """
    # Data forecasting using ARIMA
    forecasting_using_arima(data)

    return
