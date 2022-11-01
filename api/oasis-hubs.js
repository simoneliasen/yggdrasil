/* eslint-disable no-constant-condition */
/* eslint-disable no-unused-vars */
'use strict';
const request = require('superagent');
const fs = require('fs');
const admZip = require('adm-zip');

//Installation:
// download node.js
// I terminal: cd api && npm install


//Guide:        --start--      --slut--
downloadZips(2019, 7, 18, 2022, 9, 18);

//kør: cd api && node oasis-hubs.js

// den downloader 1 csv pr. hub pr. dag
// - ingen off/on-peak hubs.



async function downloadZip(startdatetime, enddatetime, i, hubName) {
    //inspiration: https://digitaldrummerj.me/node-download-zip-and-extract/

    //Create zip file folder, if it does not already exists
    const folder = "../data/market_data/zip_files"
    if (!fs.existsSync(folder)){
        fs.mkdirSync(folder);
      
        console.log('zip_files: Folder Created Successfully.');
    }

    const zipFileName = `../data/market_data/zip_files/${hubName}-${i}.zip`;

    console.log('input start date:', startdatetime);
    console.log('input end   date:', enddatetime);
    console.log('----------------');
    console.log('test start date:', '20190919T07:00-0000');
    console.log('test end   date:', '20190920T07:00-0000');



  /*  request
    .get('http://oasis.caiso.com/oasisapi/SingleZip')
    .query({queryname:'PRC_RTPD_LMP',
    startdatetime:'20190919T07:00-0000', 
    enddatetime: '20190920T07:00-0000', 
    version:1, 
    market_run_id: 'RTPD', 
    node:'TH_NP15_GEN-APND', 
    resultformat:6
        }) */

    request
    .get('http://oasis.caiso.com/oasisapi/SingleZip')
    .query({version:1, 
            resultformat:6, 
            queryname:'PRC_RTPD_LMP', 
            startdatetime:startdatetime, 
            enddatetime:enddatetime, 
            market_run_id:'RTPD', 
            node:hubName
        }) 
    .on('error', function(error) {
        console.log(error);
    })
    .pipe(fs.createWriteStream(zipFileName))
    .on('finish', function() {
        var zip = new admZip(zipFileName);
        zip.extractAllToAsync('../data/market_data/csv_files', true);
        console.log('unzipped.');
    });

}

function formatDate(date, hour) {
    const text = date.toISOString(); //altid zero utc time.
    const tmp1 = text.replaceAll('-', '');
    const tmp2 = tmp1.split(':')[0];
    const tmp3 = tmp2.substring(0, tmp2.length - 2);
    const t = hour < 10 ? `0${hour}` : hour;
    const tmp4 = tmp3 + t;
    const tmp5 = tmp4 + ':00-0000';
    return tmp5;
    // eksempel return: '20190919T07:00-0000'
}

async function downloadZips(startYear, startMonth, startDay, endYear, endMonth, endDay) {
    let dateToDownload = new Date(startYear, startMonth - 1, startDay); //month er zero indexed.
    dateToDownload.setTime(dateToDownload.getTime() + 1000*60*60*14); // tager kl. 14, pga. tidszoner-forskelle.
    const endDate = new Date(endYear, endMonth - 1, endDay);

    let i = 0;
    while(true) {
        if (dateToDownload.getTime() < endDate.getTime()) {
            // vi awaiter den, da vi vil tage 1 dag af gangen.
            await download24hours(dateToDownload, i);
            i++;
        } else {
            break;
        }
        
        //kør for næste dag:
        // note: en date's time er milisekunder siden 1. januar 1970.
        const oneDayInMS = 1000*60*60*24;
        const tomorrowTime = dateToDownload.getTime() + oneDayInMS;
        dateToDownload.setTime(tomorrowTime);
    }
}

function download24hours(date, i) {
	return new Promise((resolve, _) => {
        const startdatetime = formatDate(date, 0);
        const nextDay = new Date(date.getTime() + 1000*60*60*24)
        const enddatetime = formatDate(nextDay, 0)

        const hubs = ['TH_NP15_GEN-APND', 'TH_SP15_GEN-APND', 'TH_ZP26_GEN-APND']

        hubs.forEach((hubName, index) => {
            setTimeout(() => {
                downloadZip(startdatetime, enddatetime, i, hubName);
            }, index*8000);
        });

        // man må max kalde api'en 1 gang hvert 5. sekund. Runder op til 8 for at være sikker.
        // og resolve når alle timer er downloadet, så vi kan fortsætte til næste dag:
        setTimeout(() => {
            resolve();
        }, hubs.length*8000);
  });
}
