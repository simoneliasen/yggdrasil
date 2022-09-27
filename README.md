# Yggdrasil Commodities
*Repository for the project "Day-Ahead Electricity Price Forecasting on CAISO with Deep Learning"* 

## Getting "Real-Time" LMP Market Data From the OASIS API

*Install node.js*
https://nodejs.org/en/download/

**Install packages**
```
cd api
npm install
```
**Select time-range to download in oasis.js on line 16:**
```
downloadZips(2022, 09, 22, 2022, 09, 23);
```
**Or just write in:**
```
downloadYesterday();
```
**Then enter the following in your terminal, to run the script**:
```
node oasis.js
```

## Data Wrangling API Data, Into Model-input data.
1. Upload files gotten from the API to google drive (tested with a weeks data)
2. Open the code from data_wrangling.ipynb in your Google collab
2. Get id of of folder where your files are
- Right-click the folder
- Click Get Link
- Select substring of the link i.e. "1Fabz0Hz7xSzLZwHFpPrSaxNw4cmPq0vn"
3. Insert the Folder id in code block number 3: 
- fileList = drive.ListFile({'q': "'INSERT-ID-OF-FOLDER-HERE' in parents and trashed=false"}).GetList()
3. Run all code-blocks 
- when running the first code-block, press allow for colab to get access to your google drive.

*All generated files should be avaliable to you in the left sidepanel, by clicking the folder icon.*
