import numpy as np
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from util import TILES

CHROMEDRIVER_DIR = 'venv/bin/chromedriver'

def read_screen(driver):
    """Reads the 2048 board.

    Takes a screenshot of the browser window and gets the values
    in the 2048 board.

    Args:
        driver: webdriver
    Returns:
        grid: a 4x4 list containing the 2048 grid
    """
    image_dir = '2048.png'
    screenshot = driver.get_screenshot_as_png()
    outf = open(image_dir, 'w')
    outf.write(screenshot)
    outf.close()

    im = Image.open(image_dir)
    image = im.load()

    grid = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for i in range (4):
        for j in range (4):
            x = 590 + 240*i
            y = 590 + 240*j
            grid[j][i] = image[x,y]
   
    for i in range (4):
        grid[i] = map(lambda x: TILES[x], grid[i])

    return grid


driver = webdriver.Chrome(CHROMEDRIVER_DIR)  
driver.delete_all_cookies()
driver.get("http://gabrielecirulli.github.io/2048/")

#disable animation
with open("without-animation.js", "r") as myfile: 
    data = myfile.read().replace('\n', '')
driver.execute_script(data)

restart = driver.find_element_by_class_name("restart-button")
restart.click()

driver.quit()
