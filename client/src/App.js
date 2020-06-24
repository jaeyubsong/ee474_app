import React, { useState, useEffect } from 'react';
import axios from 'axios'
import qs from 'qs';
import './App.css';
import { Switch } from './components/Switch/Switch'
import { Flex } from './components/Flex/Flex'
axios.defaults.headers.post['Content-Type'] ='application/x-www-form-urlencoded';

const emotion = ["sadness", "fear", "neutral", "surprise", "disgust", "happiness", "anger"]
function numToEmotion (num) {
  if (num == 0) {
    return "None";
  }
  else {
    return emotion[num-1];
  }
}


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


const getMyEmotionRequest = async () => {
  const result = await axios({
    method: 'post',
    url: '/myEmotion',
    headers: {
      "Access-Control-Allow-Origin": "*",
      'content-type': 'application/x-www-form-urlencoded',
      "crossorigin": true,
      'Access-Control-Allow-Methods': 'GET, PUT, POST, DELETE, OPTIONS',
    },
    // data: qs.stringify(data)
  });
  console.log("Data received:");
  console.log(result);
  console.log(result.data.myEmotion);
  return result.data.myEmotion;
}


// emotion = [SADNESS, FEAR, NEUTRAL, SURPRISE, DISGUST, HAPPINESS, ANGER]


function App() {
  const [showMask, setShowMask] = useState(false);
  const [funMode, setFunMode] = useState(false);
  const [myEmotion, setMyEmotion] = useState(0);
  const [maskType, setMaskType] = useState(0);

  useEffect(() => {
    // showMaskRequest(showMask);
    userButtonRequest(showMask, funMode);
    const interval = setInterval(async () => {
      console.log("This will reun every second!'");
      let curEmotion = await getMyEmotionRequest();
      setMyEmotion(curEmotion);
    }, 100);
    return () => clearInterval(interval);
  })


  numToEmotion = (num) => {
    if (num == 0) {
      return "None";
    }
    else {
      return emotion[num-1];
    }
  }

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
          <h5>myEmotion is {numToEmotion(myEmotion)}</h5>

      </div>
    </div>
  );
}

export default App;
