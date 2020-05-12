"""
    File 'modeling.py' used for forecasting.
"""
import pandas as pd
import statsmodels.tsa.arima_model
import numpy as np
import data_management
import plotting


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


def modeling_applying(data):
    """
        Method to apply modeling.
        param:
            data - pandas DataFrame of initial data.
    """
    # Data forecasting using ARIMA
    forecasting_using_arima(data)

    return
