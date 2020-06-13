import React, { useState, useEffect } from 'react';
import axios from 'axios'
import qs from 'qs';
import './App.css';

axios.defaults.headers.post['Content-Type'] ='application/x-www-form-urlencoded';


const showMaskRequest = async (showMask) => {
  const data = {'showMask': showMask};
  console.log("Data sent:");
  console.log(data);
  const result = await axios({
    method: 'post',
    url: '/showMask',
    headers: {
      "Access-Control-Allow-Origin": "*",
      'content-type': 'application/x-www-form-urlencoded',
      "crossorigin": true,
      'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS',
    },
    data: qs.stringify(data)
  });
  console.log("Data received:");
  console.log(result);
}


function App() {
  const [showMask, setshowMask] = useState(false);
  const [maskType, setMaskType] = useState(0);

  useEffect(() => {
    showMaskRequest(showMask);
  })


  return (
    <div className="App">
      <div>Streaming is done below</div>
      <img src={'/stream'} />
      <button onClick={() => {
        console.log("Call showMaskRequest");
        if (showMask == false) {
          setshowMask(true);
          // showMaskRequest(true)
          console.log("showMask set to true");
        }
        else {
          setshowMask(false);
          // showMaskRequest(false)
          console.log("showMask set to false");
        }
        // showMaskRequest(showMask)
      }}>
        On/Off
      </button>
      <div>
        showMask is {showMask == true ? 'True' : 'False'}
      </div>
    </div>
  );
}

export default App;
