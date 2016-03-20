# 2048-ql

Implementation of a 2048 AI using Q-Learning and following an algorithm very similar to the one described in [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf). Currently, the AI plays better than random moves but can't make it to 1024 :(.

## Getting Started

### Set up virtualenv 
(`pip install virtualenv` if you don't have it)
```
cd 2048-ql/
virtualenv venv
source venv/bin/activate
```

### Installing python dependencies
```
(venv)$ pip install -r requirements.txt
```

### Installing Chromedriver
You can download Chromedriver here: https://sites.google.com/a/chromium.org/chromedriver/
After unzipping the folder, there will be an executable `chromedriver` (OS X). Place `chromedriver` in `venv/bin/`. 
Chromedriver should be located in `venv/bin/chromedriver`

If you aren't using OS X, place the equivalent executable in `venv/bin/` and change the `CHROMEDRIVER_DIR` variable in `main.py` to the appropriate directory.

### Train the Model
Train the model using Selenium and Chromedriver:
```
(venv)$ python main.py
```

Train the model locally, using my Python implementation of 2048:
```
(venv)$ python main.py -t local
```

## Credits
Credits to Mikhail Sannikov's [2048 bot](https://github.com/Atarity/2048-solver-bot) for `without-animation.js` and inspiration to use [Selenium](http://selenium-python.readthedocs.org/).
