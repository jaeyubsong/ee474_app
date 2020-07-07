# ee474_app

- EVA: Emotion-based Video-conferencing App
- Application for ee474 project
- Environment created by conda


## Getting Started

Let's set up the environment

### Prerequisites

- anaconda (version 4.8.3 is used)
- npm (version 6.2.0 is used)
- nodejs (version 10.8.0 is used)

### Setting up the environment
- Setup anaconda environment
```
$ conda env create -f environment.yml
```

- If creating the environment with environment.yml does not work, try the following
```
$ conda create --name ee474_app python=3.7
$ conda install -c conda-forge opencv=4.2.0=py37_7
$ conda install -c conda-forge dlib
$ conda install -c anaconda flask
$ conda install -c anaconda pillow
$ pip install pyscreenshot
```

- Then, go to client folder and setup npm
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
