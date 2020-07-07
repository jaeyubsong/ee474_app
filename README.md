# ee474_app

- EVA: Emotion-based Video-conferencing App
- Application for ee474 project
- Environment created by conda

Demo is available at [demo video](https://www.youtube.com/watch?v=uVZ_6LMGjxI&feature=youtu.be)


## Getting Started

### Prerequisites

- npm (version 6.2.0 is used)
- nodejs (version 10.8.0 is used)


- We recommend using anaconda to set up the environment
- In order to start the app,
```
$ cd client
$ npm install
```

### Download dlib model
- Download dlib model and extract the file in the project directory
```
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
$ bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

### In order to start the app
- First, turn on the backend first
```
$ cd backend
$ python app.py
```
- Then, turn on the frontend
```
$ cd frontend
$ npm start
```
- Finally, access the frontend in http://localhost:3007

- Flask backend: http://localhost:5007
- React frontend: http://localhost:3007
