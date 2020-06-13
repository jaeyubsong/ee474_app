# ee474_app

- EVA: Emotion-based Video-conferencing App
- Application for ee474 project
- Environment created by conda


## Getting Started

Let's set up the environment

### Prerequisites

- anaconda
- npm

### Setting up the environment
- Setup anaconda environment
```
$ conda env create -f environment.yml
```

### Download dlib model
- Download dlib model and extract the file in the project directory
```
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
$ bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

### In order to start the app
- First, turn on the server first
```
$ cd server
$ python app.py
```
- Then, turn on the client
```
$ cd client
$ npm start
```
- Finally, access the client in http://localhost:3007

- Flask server: http://localhost:5007
- React client: http://localhost:3007
