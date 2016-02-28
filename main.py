import model
import numpy as np
import tensorflow as tf
import time

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from util import TILES, flatten, normalize, convert_to_board


CHROMEDRIVER_DIR = 'venv/bin/chromedriver'

# setup
driver = webdriver.Chrome(CHROMEDRIVER_DIR)
inpt = tf.placeholder(tf.float32, [None, 16])
q_network = model.network(inpt)
session = tf.Session()

# using monitor with resolution 2560 x 1440
window_size = 1280
left_corner = (802, 582)
block_size = 242
# left_corner = (401, 292)
# block_size = 121

def read_screen_with_image():
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
            grid[j][i] = image[x,y][0:3]
   
    for i in range (4):
        grid[i] = map(lambda x: TILES[x], grid[i])

    return grid

def read_screen():
     """Reads the 2048 board.

    Reads screen by parsing the div classes containing the tiles.
    Much cleaner than read_screen_with_image().

    Returns:
        grid: a 4x4 list containing the 2048 grid
    """
    tile_container = driver.find_element_by_class_name('tile-container')
    divs = tile_container.find_elements_by_tag_name('div')
    grid_classes = []
    for div in divs:
        grid_classes.append(div.get_attribute('class').split(" "))

    grid = convert_to_board(grid_classes)

    return grid

def init():
    driver.delete_all_cookies()
    driver.get("http://gabrielecirulli.github.io/2048/")

    #disable animation
    with open("without-animation.js", "r") as myfile: 
        data = myfile.read().replace('\n', '')
    driver.execute_script(data)

def train():
    experiences = []
    element = driver.find_element_by_tag_name("body")
    score = 0
    epochs = 10000
    for c in range (epochs):
        board = read_screen()
       
        normalized_board = normalize(board)
        inpt_board = flatten(normalized_board)
        feed_dict = {inpt: inpt_board}
        output = model.feed_forward(q_network, feed_dict, session)

        actions = [i[0] for i in sorted(enumerate(output[0]), key=lambda x:x[1])]
        action = -1
        for i in range (3, -1, -1):
            move(element, actions[i])
            new_board = read_screen()
            if new_board != board:
                action = actions[i]
                break

        time.sleep(.2) # delay due to score update
        score_text = driver.find_element_by_class_name('score-container').text
        new_score = int(score_text.split("\n")[0])
        reward = new_score - score
        score = new_score

        experience = [board, action, reward, new_board]
        experiences.append(experience)
        

def move(element, action):
    if action == 0:
        element.send_keys(Keys.ARROW_UP)
    elif action == 1:
        element.send_keys(Keys.ARROW_RIGHT)
    elif action == 2:
        element.send_keys(Keys.ARROW_DOWN)
    elif action == 3:
        element.send_keys(Keys.ARROW_LEFT)

   
# main
driver.set_window_size(window_size, window_size)
session.run(tf.initialize_all_variables())
init()

restart = driver.find_element_by_class_name("restart-button")
restart.click()

train()

driver.quit()