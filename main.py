import argparse
import model
import numpy as np
import os
import random
import tensorflow as tf
import time
import re

from game import Game
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException
from util import TILES, flatten, normalize

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='specify web or local training (default web)')
args = parser.parse_args()

CHROMEDRIVER_DIR = 'venv/bin/chromedriver'
MODEL_PATH = os.getcwd() + "/model.ckpt"
LOCAL = 'local'
WEB = 'web'

# setup
random.seed(12345)
inpt = tf.placeholder(tf.float32, [None, 16])
q_network = model.network(inpt)
sess = tf.Session()
saver = tf.train.Saver()

driver = None
env = LOCAL if args.train == LOCAL else WEB
if env == WEB:
    driver = webdriver.Chrome(CHROMEDRIVER_DIR)

# using monitor with resolution 2560 x 1440
window_size = 1280
left_corner = (802, 582)
block_size = 242
# left_corner = (401, 292)
# block_size = 121
# retry_button_class = 'retry-button'
# keep_going_button_class = 'keep-playing-button'


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
    attempts = 5
    divs = []
    while attempts > 0:
        attempts -= 1
        try:
            divs = tile_container.find_elements_by_tag_name('div')
            break
        except:
            print "stale element, trying again"
            
    grid_classes = []
    for div in divs:
        grid_classes.append(div.get_attribute('class').split(" "))

    grid = [[0, 0, 0, 0] for i in range (4)]

    tile_regex = '^tile\-[0-9]+$'
    position_regex = '^tile\-position\-.+$'

    for grid_class in grid_classes:
        tile_class = filter(lambda x: re.match(tile_regex, x), grid_class)
        position_class = filter(lambda x: re.match(position_regex, x), grid_class)

        if tile_class == [] or position_class == []:
            continue

        tile_value = int(tile_class[0].split("-")[-1])
        pos_split = position_class[0].split("-")
        col = int(pos_split[-2]) - 1
        row = int(pos_split[-1]) - 1

        grid[row][col] = tile_value

    return grid

def get_board(env, game=None):
    if env == LOCAL:
        return game.grid
    else:
        return read_screen()

def init():
    driver.delete_all_cookies()
    driver.get("http://gabrielecirulli.github.io/2048/")

    #disable animation
    with open("without-animation.js", "r") as myfile: 
        data = myfile.read().replace('\n', '')
    driver.execute_script(data)

def train(env):
    # Training parameters
    batch_size = 2
    discount = 0.9
    epochs = 10000
    epsilon = 1

    # Experience Replay Memory
    experiences = []

    # Setup
    game = None
    if env == LOCAL:
        game = Game()
        game.add_random_tile()

    score = 0
    board = get_board(env, game)
    inpt_board = flatten(normalize(board))

    for c in range (epochs):
        if env == WEB:
            game = driver.find_element_by_tag_name("body")
        feed_dict = {inpt: inpt_board.reshape((1,16))}

        action_indices = []
        action = -1

        # Epsilon-greedy action selection
        epsilon = max(1 - float(c) / epochs * 2, 0.1)
        rand = random.random()
        if rand < epsilon:
            action_indices = [0,1,2,3]
            random.shuffle(action_indices)
        else:
            values = model.feed_forward(q_network, feed_dict, sess)
            action_indices = [i[0] for i in sorted(enumerate(values[0]), key=lambda x:x[1], reverse=True)]

        for i in range (4):
            move(env, game, action_indices[i])
            time.sleep(.1)
            new_board = get_board(env, game)
            if new_board != board:
                action = action_indices[i]
                if env == LOCAL:
                    game.display()
                    game.add_random_tile()
                break

        # if none of the actions are valid, restart the game
        if action == -1:
            driver.find_element_by_class_name("restart-button").click()

        if env == WEB:
            score_text = driver.find_element_by_class_name('score-container').text
            new_score = int(score_text.split("\n")[0])
        else:
            new_score = game.score

        reward = new_score - score
        score = new_score
        next_inpt_board = flatten(normalize(new_board))

        # Save experience
        one_hot_reward = np.zeros((4))
        one_hot_reward[action] = reward
        experience = [inpt_board, one_hot_reward, next_inpt_board]
        experiences.append(experience)

        # Update board
        board = new_board
        inpt_board = next_inpt_board
        
        # Train the model using experience replay
        if c > 0 and c % batch_size == 0 and batch_size < len(experiences):
            sample = random.sample(experiences, batch_size)
            last_states = np.array(map(lambda x: x[2], sample))
            rewards = np.array(map(lambda x: x[1], sample))

            feed_dict = {inpt: last_states}
            values = model.feed_forward(q_network, feed_dict, sess)

            y = np.add(rewards, discount * values)
            loss = tf.reduce_sum(tf.square(y - q_network))
            grad_op = model.train(loss)
            save_path = saver.save(sess, MODEL_PATH)


def move(env, game, action):
    if env == LOCAL:
        game.move(action)
    else:
        if action == 0:
            game.send_keys(Keys.ARROW_UP)
        elif action == 1:
            game.send_keys(Keys.ARROW_RIGHT)
        elif action == 2:
            game.send_keys(Keys.ARROW_DOWN)
        elif action == 3:
            game.send_keys(Keys.ARROW_LEFT)


# main
if os.path.isfile(MODEL_PATH):
    saver.restore(sess, MODEL_PATH)
else:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

if env == WEB:
    init()
    restart = driver.find_element_by_class_name("restart-button")
    restart.click()
    time.sleep(0.5) # allow game to load

train(env)

if env == WEB:
    driver.quit()