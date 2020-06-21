library(forecast)
library(zoo)

data <- read.csv("C:/Users/Admin/PycharmProjects/Epileptic_Seizure_Recognition/test.csv")[,2]
data <- ts(data, start=0, frequency = 178)
data <- rollmedian(data, 5)

modelArima = Arima(data, order = c(50,1,80), method = "CSS")
summary(modelArima)
forecastResults <- forecast(modelArima, h = 25)$mean
plot(data, 
     main = "Brain activity of patient 791",
     xlab = "Time, s",
     ylab = "Brain activity",
     col = "blue",
     xlim = c(0, 1.15))
lines(forecastResults, col="red")
legend("topright", c("Observations", "Forecast"),  fill = c("blue", "red"))

trainingSet2 <- head(data, 140)
validationSet2 <- tail(data, 34)

modelArima = Arima(head(data, 116), order = c(30,2,80), method = "CSS")
summary(modelArima)
forecastResults <- forecast(modelArima, h = 58)$mean

nrmse <- sqrt(mean((tail(data, 58) - forecastResults) ^ 2)) / 
  mean(validationSet1)
nrmse
mae <- mean(abs(tail(data, 58) - forecastResults))
mae
mse <- mean((tail(data, 58) - forecastResults) ^ 2)
mse
plot(data, 
     main = "Brain activity of patient 791 (nRMSE = 17.03, MAE = 70.33, MSE = 7058.96)",
     xlab = "Time, s",
     ylab = "Brain activity",
     col = "blue",
     xlim = c(0, 1.15))
lines(forecastResults, col="red")
legend("topright", c("Observations", "Forecast"),  
       fill = c("blue", "red"))


modelArima = Arima(head(data, 140), order = c(50,1,80), method = "CSS")
summary(modelArima)
forecastResults <- forecast(modelArima, h = 34)$mean

nrmse <- sqrt(mean((tail(data, 34) - forecastResults) ^ 2)) / 
  mean(validationSet1)
nrmse
mae <- mean(abs(tail(data, 34) - forecastResults))
mae
mse <- mean((tail(data, 34) - forecastResults) ^ 2)
mse
plot(data, 
     main = "Brain activity of patient 791 (nRMSE = 8.30, MAE = 31.99, MSE = 1678.49)",
     xlab = "Time, s",
     ylab = "Brain activity",
     col = "blue",
     xlim = c(0, 1.15))
lines(forecastResults, col="red")
legend("topright", c("Observations", "Forecast"),  
       fill = c("blue", "red"))

