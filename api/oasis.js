/* eslint-disable no-constant-condition */
/* eslint-disable no-unused-vars */
'use strict';
const request = require('superagent');
const fs = require('fs');

//Installation:
// download node
// I terminal: npm install


//Guide:          --start--      --slut--
// downloadZips(2022, 09, 22, 2022, 09, 23); //dato i vores tid! - ikke sikkert det er samme dato california.
//eller:
downloadYesterday();

//kør: cd api
// node oasis

// den har issues ved at downloade - end point er vidst rigtig tho.
// TODO: UTC ting skal fixes. ELLER MÅSKE BARE SÆT TXX MANUELT.ß
// Extract zip files i "csv" folder.



async function downloadZip(startdatetime, groupId = 'RTPD_LMP_GRP') {
    //inspiration: https://digitaldrummerj.me/node-download-zip-and-extract/ 
    const baseUrl = 'http://oasis.caiso.com/oasisapi/GroupZip?version=1&resultformat=6';
    const urlWithParams = `${baseUrl}&groupid=${groupId}&startdatetime=${startdatetime}`;
    console.log('urlWithParams:', urlWithParams);

    const hourForFileName = startdatetime.split(':')[0].split('T')[1];
    const zipFileName = `./zip_files/${groupId}-T${hourForFileName}.zip`;

    request
    .get(urlWithParams)
    .on('error', function(error) {
        console.log(error);
    })
    .pipe(fs.createWriteStream(zipFileName))
    .on('finish', function() {
        // add code below to here
    });

}

function formatDate(date, opr_hour) {
    const text = date.toISOString(); //altid zero utc time.
    const tmp1 = text.replaceAll('-', '');
    const tmp2 = tmp1.split(':')[0];
    const tmp3 = tmp2.substring(0, tmp2.length - 2);
    const t = opr_hour < 10 ? `0${opr_hour}` : opr_hour;
    const tmp4 = tmp3 + t;
    const tmp5 = tmp4 + ':00-0000';
    return tmp5;
    // eksempel return: '20190919T07:00-0000'
}

async function downloadZips(startYear, startMonth, startDay, endYear, endMonth, endDay) {
    let dateToDownload = new Date(startYear, startMonth - 1, startDay); //month er zero indexed.
    const endDate = new Date(endYear, endMonth - 1, endDay);
    while(true) {
        if (dateToDownload.getTime() < endDate.getTime()) {
            download24hours(dateToDownload);
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
    const oneDayInMS = 1000*60*60*24;
    const yesterday = new Date(today.getTime() - oneDayInMS);

    downloadZips(yesterday.getFullYear(), yesterday.getMonth() + 1, yesterday.getDate(), today.getFullYear(), today.getMonth() + 1, today.getDate());
}

async function download24hours(date) {
    for (let i = 1; i < 25; i++) {
        let startdatetime = formatDate(date, i);
        downloadZip(startdatetime);
    }
}