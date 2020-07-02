import React, { useState, useEffect } from 'react';
import axios from 'axios'
import qs from 'qs';
import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import FormLabel from '@material-ui/core/FormLabel';
import {RadialChart} from 'react-vis';
import './App.css';
import { Switch } from './components/Switch/Switch'
import { Flex } from './components/Flex/Flex'
axios.defaults.headers.post['Content-Type'] ='application/x-www-form-urlencoded';

// astonished 
// unsatisfied
// joyful
// neutral
// sad

// const myData = [{
//   angle: 3,
//   label: 'astonished'
// }, 
//   {
//   angle: 2,
//   label: 'unsatisfied'
// }, 
// {
//   angle: 1,
//   label: 'joyful'
// },
// {
//   angle: 1,
//   label: 'neutral'
// },
// {
//   angle: 1,
//   label: 'sad'
// }]

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

const setEmotionRequest = async (myEmotion) => {
  const data = {'emotion': myEmotion};
  console.log("Data sent:");
  console.log(data);
  const result = await axios({
    method: 'post',
    url: '/setMyEmotion',
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

// const changeStatGraph = (astonished, unsatisfied, joyful, neutral, sad) => {
//     myData[0].angle=astonished
//     myData[1].angle=unsatisfied
//     myData[2].angle=joyful
//     myData[3].angle=neutral
//     myData[4].angle=sad
//   } 


// emotion = [SADNESS, FEAR, NEUTRAL, SURPRISE, DISGUST, HAPPINESS, ANGER]
let update_stat_interval = 0;
let update_scene_interval = 0;
let scene_cur_index = 0;


function App() {
  const [showMask, setShowMask] = useState(false);
  const [funMode, setFunMode] = useState(false);
  const [myEmotion, setMyEmotion] = useState(0);
  const [myEmoji, setMyEmoji] = useState([
    './emoji/neutral/neutral_e(1).png', 
    './emoji/neutral/netural_m(1).png',
    './emoji/neutral/neutral(1).png'
  ]);
  const [curImageShow, setCurImageShow] = useState(0)
  const [curImageShowPath, setCurImageShowPath] = useState([0])
  const [maskType, setMaskType] = useState("7");
  const [emotionData, setEmotionData] = useState(
    [{
      angle: 0,
      label: 'astonished'
    }, 
      {
      angle: 0,
      label: 'unsatisfied'
    }, 
    {
      angle: 0,
      label: 'joyful'
    },
    {
      angle: 100,
      label: 'neutral'
    },
    {
      angle: 0,
      label: 'sad'
    }]);
  const [startVideo, setStartVideo] = useState(false);





  useEffect(() => {
    // showMaskRequest(showMask);
    userButtonRequest(showMask, funMode, maskType);
    const interval = setInterval(async () => {
      if (startVideo) {
        if (update_stat_interval == 20) { // 20
          let astonished_angle = 0
          let unsatisfied_angle = 0
          let joyful_angle = 0
          let neutral_angle = 100
          let sad_angle = 0
          if (scene_cur_index == 0) {
            astonished_angle = 0
            unsatisfied_angle = 0
            joyful_angle = 0
            neutral_angle = 100
            sad_angle = 0
          }
          else if (scene_cur_index == 1) { //40
            astonished_angle = 0
            unsatisfied_angle = 16
            joyful_angle = 0
            neutral_angle = 33
            sad_angle = 50
          }
          else if (scene_cur_index == 2) { //60
            astonished_angle = 0
            unsatisfied_angle = 0
            joyful_angle = 33
            neutral_angle = 50
            sad_angle = 16
          }
          else if (scene_cur_index == 3) { //80
            astonished_angle = 0
            unsatisfied_angle = 0
            joyful_angle = 16
            neutral_angle = 33
            sad_angle = 50
          }
          else if (scene_cur_index == 4) { //100
            astonished_angle = 0
            unsatisfied_angle = 0
            joyful_angle = 0
            neutral_angle = 33
            sad_angle = 66
          }
          else {
            astonished_angle = 35
            unsatisfied_angle = 25
            joyful_angle = 20
            neutral_angle = 15
            sad_angle = 5
          }
          changeEmotionStat(astonished_angle, unsatisfied_angle, joyful_angle, neutral_angle, sad_angle)
          update_stat_interval = 0
          scene_cur_index = scene_cur_index + 1
        }
        else{
          update_stat_interval = update_stat_interval + 1
        }
  
        if (update_scene_interval == 5) {
          let imageNow = curImageShow + 5
          setCurImageShow(imageNow)
          update_scene_interval = 0
        }
        else {
          update_scene_interval =  update_scene_interval + 1
        }
      }

      // changeStatGraph(astonished_angle, unsatisfied_angle, joyful_angle, neutral_angle, sad_angle)
      console.log("This will reun every second!'");
      let curEmotion = await getServerDataRequest()
      // let curEmotion = await setEmotionRequest(myEmotion);
      setMyEmotion(curEmotion);


    }, 200);
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

  const changeEmotionStat = (astonished_angle, unsatisfied_angle, joyful_angle, neutral_angle, sad_angle) => {

    const newStat = [{
      angle: astonished_angle,
      label: 'astonished'
    }, 
      {
      angle: unsatisfied_angle,
      label: 'unsatisfied'
    }, 
    {
      angle: joyful_angle,
      label: 'joyful'
    },
    {
      angle: neutral_angle,
      label: 'neutral'
    },
    {
      angle: sad_angle,
      label: 'sad'
    }]
    setEmotionData(newStat)
  
  }



  const emoji_recommend_one_one = (index) => {
    switch (index) {
      case 1: return require('./emoji/astonished/astonished_e(1).png');
      case 2: return require('./emoji/unsatisfied/unsatisfied_e(1).png');
      case 3: return require('./emoji/joyful/joyful_e(1).png');
      case 4: return require('./emoji/neutral/neutral_e(1).png');
      case 5: return require('./emoji/sad/sad_e(1).png');
      default: return require('./emoji/neutral/neutral_e(1).png');
      }
  }

  const emoji_recommend_two_one = (index) => {
    switch (index) {
      case 1: return require('./emoji/astonished/astonished_m(1).png');
      case 2: return require('./emoji/unsatisfied/unsatisfied_m(1).png');
      case 3: return require('./emoji/joyful/joyful_m(1).png');
      case 4: return require('./emoji/neutral/neutral_m(1).png');
      case 5: return require('./emoji/sad/sad_m(1).png');
      default: return require('./emoji/neutral/neutral_m(1).png');
      }
  }

  const emoji_recommend_three_one = (index) => {
    switch (index) {
      case 1: return require('./emoji/astonished/astonished(1).png');
      case 2: return require('./emoji/unsatisfied/unsatisfied(1).png');
      case 3: return require('./emoji/joyful/joyful(1).png');
      case 4: return require('./emoji/neutral/neutral(1).png');
      case 5: return require('./emoji/sad/sad(1).png');
      default: return require('./emoji/neutral/neutral(1).png');
      }
  }


  return (
    <div className="App">
      {/* div for user */}
      {/* <Flex container width=> */}
      <Flex container width="100%" margin="0 0 0 15px">
          <Flex flex={0.3}>
            <img src={'/stream'} width="500px" style={{marginTop: '5px'}}/>
            <h2>My current emotion is {numToEmotion(myEmotion)}</h2>

            <Flex container justifyContent="flex-start" margin="10px 0 0 25px">
              <h4 style={{marginTop: '0px'}}>ShowMask</h4>
              <Switch
                isOn={showMask}
                id={"showMask"}
                handleToggle={() => setShowMask(!showMask)}
            ></Switch>
            </Flex>

            <Flex container justifyContent="flex-start" margin="0 0 0 25px">
              <h4 style={{marginTop: '0px'}}>Fun Mode!!</h4>
              <Switch
                isOn={funMode}
                id={"funMode"}
                handleToggle={() => setFunMode(!funMode)}
            ></Switch>
            </Flex>

            <Flex container justifyContent="flex-start" margin="0 0 0 25px">
              <FormControl component="fieldset">
                {/* <FormLabel component="legend">Choose your mask!</FormLabel> */}
                <h3 style={{marginTop: '20px', marginBottom: '0px'}}>Choose your mask!!</h3>
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
          <Flex flex={0.1} margin="0 0 0 5px">
            We recomend you the following emojis!
            <div>
                <img src={emoji_recommend_one_one(myEmotion)} width="75px"/>
            </div>
            <div>
                <img src={emoji_recommend_two_one(myEmotion)} width="75px"/>
            </div>
            <div>
                <img src={emoji_recommend_three_one(myEmotion)} width="75px"/>
            </div>
            
          </Flex>
          <Flex flex={0.6} margin="0 0 0 5px">
          imgSource
            {/* <img src={require('./gender/0.png')} width="800px" style={{marginTop: '5px', marginBottom: '-55px'}}/> */}
            {/* <img src={imgSource(curImageShow)} width="800px" style={{marginTop: '5px', marginBottom: '-55px'}}/> */}
            <Flex container justifyContent="justify-content">

              <Flex flex={0.4} margin="0 0 0 5px">
                <RadialChart
                    data={emotionData}
                    // showLabels={true}
                    labelsRadiusMultiplier={0.9}
                    margin={{left: 40, right: 10, top: 10, bottom: 10}}
                    width={400}
                    height={500} />
              </Flex>

              <Flex flex={0.6}>
                <h5 style={{marginLeft: '0px', marginTop: '55px'}}>
                <img src={require('./astonished.png')} width="20px"/>
                  Astonished={emotionData[0].angle}%
                <img src={require('./unsatisfied.png')} width="20px"/>  
                  Unsatisfied={emotionData[1].angle}%
                <img src={require('./joyful.png')} width="20px"/>
                  Joyful={emotionData[2].angle}%
                <img src={require('./neutral.png')} width="20px"/>
                  Neutral={emotionData[3].angle}%
                <img src={require('./sad.png')} width="20px"/>
                Sad={emotionData[4].angle}%
                </h5>
                <h5 style={{marginLeft: '0px', marginTop: 'px'}}>
                  Sleepy audience: Undefined for now
                </h5>
              </Flex>

            </Flex>

          </Flex> 

      </Flex>

      <div>
           
      </div>
      <div>
{/* 
          <h5>showMask is {showMask == true ? 'True' : 'False'}</h5>
          <h5>funMode is {funMode == true ? 'True' : 'False'}</h5>
          <h5>myEmotion is {numToEmotion(myEmotion)}</h5> */}

      </div>
    </div>
  );
}

export default App;
