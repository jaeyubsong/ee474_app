# ee474_server

- Local server for ee474 project
- Environment created by conda


## Getting Started

Let's set up the environment

### Prerequisites

- anaconda

### Setting up the environment
- Setup anaconda environment
```
$ cd server
$ conda env create -f environment.yml
```

### Download dlib model
- Download dlib model and extract the file in the project directory
```
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
$ bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

- Flask server: http://localhost:5007
