/* eslint-disable no-constant-condition */
/* eslint-disable no-unused-vars */
'use strict';
const request = require('superagent');
const fs = require('fs');
const admZip = require('adm-zip');

//Installation:
// download node.js
// I terminal: cd api && npm install


//Guide:          --start--      --slut--
downloadZips(2022, 7, 1, 2022, 8, 1);
//eller:

//2022-08-01-2022-08-08
//downloadYesterday();

//kør: cd api && node oasis.js
// (den tager ca. 4 min pr. dag der skal hentes.)



async function downloadZip(startdatetime, groupId = 'RTPD_LMP_GRP') {
    //inspiration: https://digitaldrummerj.me/node-download-zip-and-extract/ 
    const baseUrl = 'http://oasis.caiso.com/oasisapi/GroupZip?version=1&resultformat=6';
    const urlWithParams = `${baseUrl}&groupid=${groupId}&startdatetime=${startdatetime}`;
    console.log('downloader:', urlWithParams);

    //Create zip file folder, if it does not already exists
    const folder = "./zip_files"
    if (!fs.existsSync(folder)){
        fs.mkdirSync(folder);
      
        console.log('zip_files: Folder Created Successfully.');
    }

    const hourForFileName = startdatetime.split(':')[0].split('T')[1];
    const zipFileName = `./zip_files/${groupId}-T${hourForFileName}.zip`;

    request
    .get(urlWithParams)
    .on('error', function(error) {
        console.log(error);
    })
    .pipe(fs.createWriteStream(zipFileName))
    .on('finish', function() {
        var zip = new admZip(zipFileName);
        zip.extractAllToAsync('./csv_files', true);
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
    while(true) {
        if (dateToDownload.getTime() < endDate.getTime()) {
            // vi awaiter den, da vi vil tage 1 dag af gangen.
            await download24hours(dateToDownload);
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

async function downloadYesterday() {
    const today = new Date();
    const dayInMS = 1000*60*60*24; //pga. tidszoner.
    const yesterday = new Date(today.getTime() - dayInMS);

    downloadZips(yesterday.getFullYear(), yesterday.getMonth() + 1, yesterday.getDate(), today.getFullYear(), today.getMonth() + 1, today.getDate());
}

function download24hours(date) {
	return new Promise((resolve, _) => {
        for (let i = 0; i < 24; i++) {
        // man må max kalde api'en 1 gang hvert 5. sekund. Runder op til 8 for at være sikker.
        setTimeout(() => {
            let startdatetime = formatDate(date, i);
            downloadZip(startdatetime);
        }, 8000*i);
    }

    // og resolve når alle timer er downloadet, så vi kan fortsætte til næste dag:
    setTimeout(() => {
        resolve();
    }, 26*8000);
  });
}
