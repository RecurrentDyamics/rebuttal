library(imputeTS)
library(forecast)
x <- read.csv('./missing.csv', header = TRUE)
df <- x[,]

for (i in 2: 37) {
    print(i)
    r <- na.kalman(unlist(x[,i]), model = 'auto.arima')
    df[, i] <- r
}
write.csv(df, './imputation.csv')
