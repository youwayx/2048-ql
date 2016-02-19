# 2048-ql

## Getting Started

### Set up virtualenv 
(`pip install virtualenv` if you don't have it)
```
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

If you aren't using OS X, place the equivalent executable in `venv/bin/`.

