import React, { useState, useEffect } from 'react';
import axios from 'axios'
import qs from 'qs';
import './App.css';
import { Switch } from './components/Switch/Switch'
import { Flex } from './components/Flex/Flex'
axios.defaults.headers.post['Content-Type'] ='application/x-www-form-urlencoded';


const userButtonRequest = async (showMask, funMode) => {
  const data = {'showMask': showMask, 'funMode': funMode};
  console.log("Data sent:");
  console.log(data);
  const result = await axios({
    method: 'post',
    url: '/userButton',
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
  const [showMask, setShowMask] = useState(false);
  const [funMode, setFunMode] = useState(false);
  const [maskType, setMaskType] = useState(0);

  useEffect(() => {
    // showMaskRequest(showMask);
    userButtonRequest(showMask, funMode);
  })


  return (
    <div className="App">
      <div>Streaming is done below</div>

      {/* div for user */}
      <Flex container width="500px">
          <Flex flex={0.6}>
            <img src={'/stream'} width="500px"/>
          </Flex>
      </Flex>
      <Flex flex={1} margin="0 0 0 5px">
        <Flex container justifyContent="flex-start">
            <h4 style={{marginTop: '0px'}}>ShowMask</h4>
            <Switch
              isOn={showMask}
              id={"showMask"}
              handleToggle={() => setShowMask(!showMask)}
          ></Switch>
        </Flex>

        <Flex container justifyContent="flex-start">
            <h4 style={{marginTop: '0px'}}>Fun Mode!!</h4>
            <Switch
              isOn={funMode}
              id={"funMode"}
              handleToggle={() => setFunMode(!funMode)}
          ></Switch>
        </Flex>

      </Flex>
      <div>
        
      </div>
      <div>

          <h5>showMask is {showMask == true ? 'True' : 'False'}</h5>
          <h5>funMode is {funMode == true ? 'True' : 'False'}</h5>

      </div>
    </div>
  );
}

export default App;
