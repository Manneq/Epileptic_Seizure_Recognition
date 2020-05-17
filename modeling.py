"""
    File 'modeling.py' used for forecasting and finding brain activity
        distributions over categories.
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
        plotting.line_plotting(patient_data,
                               ["Time, s", "Brain activity"],
                               "Brain of patient " + str(patient),
                               "modeling/forecasting",
                               forecasted_data=forecasted_patient_data)

    return


def distribution_thesis(data):
    """
        Method to plot mean distribution for categories.
        param:
            data - pandas DataFrame of initial data
    """
    for category in data['categories'].unique().tolist():
        # Distributions computing
        mean_distribution = \
            data.loc[data['categories'] == category].\
            drop(columns=['y', 'categories'], axis=1).mean(axis=0)

        # Distribution plotting
        plotting.line_plotting(mean_distribution,
                               ["Brain activity", "Mean"],
                               "Distribution for " + category,
                               "modeling/distributions",
                               font_size=8,
                               distribution=True)

    return


def modeling_applying(data):
    """
        Method to apply modeling.
        param:
            data - pandas DataFrame of initial data.
    """
    # Make a thesis about brain activities mean distributions
    distribution_thesis(data)

    # Data forecasting using ARIMA
    forecasting_using_arima(data)

    return
