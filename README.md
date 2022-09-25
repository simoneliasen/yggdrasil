# Yggdrasil Commodities
Repository for the project "Day-Ahead Electricity Price Forecasting on CAISO with Deep Learning"

## Getting "Real-Time" LMP Market Data

**Download node.js**
```
cd api && npm install
```
**Select time-range to download in oasis.js on line 16:**
```
downloadZips(2022, 09, 22, 2022, 09, 23);
```
**or just write in:**
```
downloadYesterday();
```
**then enter the following in your terminal, to run the script**:
```
node oasis.js
```
