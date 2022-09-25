# yggdrasil
repository for work with yggdrasil commodities

## Getting "real-time" LMP market data

###### Download node.js
```
cd api && npm install
```

######  Select time-range to download in oasis.js on line 16:
```
downloadZips(2022, 09, 22, 2022, 09, 23);
```
######  just:
```
downloadYesterday();
```
######  then enter the following in your terminal, to run the script:
```
node oasis.js
```
