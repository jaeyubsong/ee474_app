import React, { useState, useEffect } from 'react';
import axios from 'axios'
import qs from 'qs';
import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import FormLabel from '@material-ui/core/FormLabel';
import './App.css';
import { Switch } from './components/Switch/Switch'
import { Flex } from './components/Flex/Flex'
axios.defaults.headers.post['Content-Type'] ='application/x-www-form-urlencoded';

// astonished 
// unsatisfied
// joyful
// neutral
// sad
const emotion = ["astonished", "unsatisfied", "joyful", "neutral", "sad"]
const mask = ["blindFold", "bunny", "darthVadar", "grouchoGlasses", "guyFawkes", "halloween", "surgicalMask"]
function numToEmotion (num) {
  if (num == 0) {
    return "None";
  }
  else {
    return emotion[num-1];
  }
}


const userButtonRequest = async (showMask, funMode, maskType) => {
  const data = {'showMask': showMask, 'funMode': funMode, 'maskType': maskType};
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


const getServerDataRequest = async () => {
    const result = await axios({
      method: 'post',
      url: '/getServerData',
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
  const [maskType, setMaskType] = useState("7");

  useEffect(() => {
    // showMaskRequest(showMask);
    userButtonRequest(showMask, funMode, maskType);
    const interval = setInterval(async () => {
      console.log("This will reun every second!'");
      let curEmotion = await getServerDataRequest();
      setMyEmotion(curEmotion);
    }, 100);
    return () => clearInterval(interval);
  })

  const handleMaskTypeChange = (event) => {
    setMaskType(event.target.value);
  }


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
      <Flex>
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
          <Flex container justifyContent="flex-start">
            <FormControl component="fieldset">
              <FormLabel component="legend">Choose your mask!</FormLabel>
              <RadioGroup aria-label="gender" name="gender1" value={maskType} onChange={handleMaskTypeChange}>
                <FormControlLabel value="1" control={<Radio />} label="Blind Fold" />
                <FormControlLabel value="2" control={<Radio />} label="Bunny" />
                <FormControlLabel value="3" control={<Radio />} label="Darth Vadar" />
                <FormControlLabel value="4" control={<Radio />} label="Groucho Glasses" />
                <FormControlLabel value="5" control={<Radio />} label="Guy Fawkes" />
                <FormControlLabel value="6" control={<Radio />} label="Halloween" />
                <FormControlLabel value="7" control={<Radio />} label="Surgical Mask" />
                {/* <FormControlLabel value={4} disabled control={<Radio />} label="(Disabled option)" /> */}
              </RadioGroup>
            </FormControl>
          </Flex>

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
