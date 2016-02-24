import model
import numpy as np
import tensorflow as tf

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from util import TILES, flatten, normalize


CHROMEDRIVER_DIR = 'venv/bin/chromedriver'

# using monitor with resolution 2560 x 1440
window_size = 1280
left_corner = (401, 292)
block_size = 121

# setup
driver = webdriver.Chrome(CHROMEDRIVER_DIR)
inpt = tf.placeholder(tf.float32, [None, 16])
q_network = model.network(inpt)
session = tf.Session()

def read_screen():
    """Reads the 2048 board.

    Takes a screenshot of the browser window and gets the values
    in the 2048 board.

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
            x,y = left_corner
            x += block_size*i
            y += block_size*j
            grid[j][i] = image[x,y]
   
    for i in range (4):
        grid[i] = map(lambda x: TILES[x], grid[i])

    return grid

def init():
    driver.delete_all_cookies()
    driver.get("http://gabrielecirulli.github.io/2048/")

    #disable animation
    with open("without-animation.js", "r") as myfile: 
        data = myfile.read().replace('\n', '')
    driver.execute_script(data)

def play_game():
    board = read_screen()
    normalized_board = normalize(board)
    inpt_board = flatten(normalized_board)
    feed_dict = {inpt: inpt_board}
    output = model.feed_forward(q_network, feed_dict, session)
    print output

   
# main
driver.set_window_size(window_size, window_size)
session.run(tf.initialize_all_variables())
init()

restart = driver.find_element_by_class_name("restart-button")
restart.click()

play_game()

driver.quit()