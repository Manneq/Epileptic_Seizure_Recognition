library(forecast)
library(zoo)

data <- read.csv("C:/Users/Admin/PycharmProjects/Epileptic_Seizure_Recognition/test.csv")[,2]
data <- ts(data, start=0, frequency = 178)
data <- rollmedian(data, 5)

modelArima = Arima(data, order = c(30,2,100), method = "CSS")
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

trainingSet1 <- head(data, 61)
trainingSet2 <- head(data, 116)
validationSet1 <- tail(data, 113)
validationSet1 <- head(validationSet1, 61)
validationSet2 <- tail(data, 58)

modelArima = Arima(head(data, 61), order = c(25,2,30), method = "CSS")
summary(modelArima)

forecastResults <- forecast(modelArima, h = 61)$mean
nRMSE <- sqrt(mean(sum((validationSet1 - forecastResults) ^ 2))) / 
  mean(validationSet1)
nRMSE

plot(trainingSet1, 
     main = "Brain activity of patient 791",
     xlab = "Time, s",
     ylab = "Brain activity",
     col = "blue",
     xlim = c(0, 1.15))
lines(validationSet1, col="green")
lines(forecastResults, col="red")
legend("topright", c("Training set", "Validation set", "Forecast"),  
       fill = c("blue", "green", "red"))

modelArima = Arima(head(data, 116), order = c(30,2,80), method = "CSS")
summary(modelArima)
forecastResults <- forecast(modelArima, h = 58)$mean

nRMSE <- sqrt(mean(sum((validationSet2 - forecastResults) ^ 2))) / 
  mean(validationSet1)
nRMSE

plot(trainingSet2, 
     main = "Brain activity of patient 791",
     xlab = "Time, s",
     ylab = "Brain activity",
     col = "blue",
     xlim = c(0, 1.15))
lines(validationSet2, col="green")
lines(forecastResults, col="red")
legend("topright", c("Training set", "Validation set", "Forecast"),  
       fill = c("blue", "green", "red"))

