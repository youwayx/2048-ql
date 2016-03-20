import argparse
import numpy as np
import os
import pickle
import random
import tensorflow as tf
import time
import re

from game import Game
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import StaleElementReferenceException
from util import TILES, flatten, normalize, normalize_num, get_reward

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='specify web or local training (default web)')
args = parser.parse_args()

CHROMEDRIVER_DIR = 'venv/bin/chromedriver'
MODEL_PATH = os.getcwd() + "/abcd.ckpt"
EXPERIENCES_PATH = os.getcwd() + "/abcd.p"
LOCAL = 'local'
WEB = 'web'

driver = None
env = LOCAL if args.train == LOCAL else WEB
if env == WEB:
    driver = webdriver.Chrome(CHROMEDRIVER_DIR)

# retry_button_class = 'retry-button'
# keep_going_button_class = 'keep-playing-button'

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

def display_board(board):
    for i in range(4):
        print board[i]

def network():
    """Builds the neural network to approximate Q function

    Returns:
        inpt: Input layer of the network.
        out: Output layer of the network.
    """

    inpt = tf.placeholder("float", [None, 16])

    W_1 = tf.Variable(tf.truncated_normal([16, 48], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.01, shape=[48]))

    h_1 = tf.nn.relu(tf.matmul(inpt, W_1) + b_1)

    W_2 = tf.Variable(tf.truncated_normal([48, 16], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.01, shape=[16]))

    h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)

    W_3 = tf.Variable(tf.truncated_normal([16, 4], stddev=0.1))
    b_3 = tf.Variable(tf.constant(0.01, shape=[4]))

    out = tf.nn.relu(tf.matmul(h_2, W_3) + b_3)

    return inpt, out

def init():
    driver.delete_all_cookies()
    driver.get("http://gabrielecirulli.github.io/2048/")

    #disable animation
    with open("without-animation.js", "r") as myfile: 
        data = myfile.read().replace('\n', '')
    driver.execute_script(data)

def train(inpt, out, env, sess):
    # Training parameters
    batch_size = 32
    discount = 0.9
    epochs = 5000000
    epsilon = 1
    learning_rate = 1e-6
    start = 0
    num_explore = 100000
    max_exps = 30000

    # Loss Function
    y = tf.placeholder("float", [None, 4])
    loss = tf.reduce_sum(tf.square(y - out))
    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()
    if os.path.isfile(MODEL_PATH):
        print "Loading model."
        saver.restore(sess, MODEL_PATH)
        start = num_explore + max_exps
    else:
        print "Initializing new model."
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

    # Experience Replay Memory
    data = {}
    experiences = []
    avg_scores = []
    epochs_trained = 0
    index = 0
    if os.path.isfile(EXPERIENCES_PATH):
        data = pickle.load(open(EXPERIENCES_PATH, 'rb'))
        experiences = data['experiences']
        epochs_trained = data['epochs_trained']
        avg_scores = data['avg_scores']
        max_exps = len(experiences)
    else:
        data['experiences'] = []
        data['epochs_trained'] = 0
        data['avg_scores'] = []

    # Setup
    game = None
    if env == LOCAL:
        game = Game()
        game.add_random_tile()

    score = 0
    board = get_board(env, game)
    inpt_board = flatten(normalize(board))
    prev_inpt_board = None
    prev_action = -1

    tot_score_counter = 0
    game_counter = 0
    avg_score = 0
    for c in range (start, epochs):
        if env == WEB:
            game = driver.find_element_by_tag_name("body")
        feed_dict = {inpt: inpt_board.reshape((1,16))}

        action_indices = []
        action = -1

        # Epsilon-greedy action selection
        iterations = max(0, c - max_exps)
        epsilon = max(1 - float(iterations) / num_explore, 0.01)
        rand = random.random()
        if rand < epsilon:
            action_indices = [0,1,2,3]
            random.shuffle(action_indices)
        else:
            values = out.eval(feed_dict=feed_dict, session=sess)
            action_indices = [i[0] for i in sorted(enumerate(values[0]), key=lambda x:x[1], reverse=True)]

        action, new_board = move(env, game, action_indices, board)

        # if none of the actions are valid, restart the game
        if action == -1:
            if prev_inpt_board is not None:
                one_hot_reward = np.zeros((4))
                one_hot_reward[prev_action] = -1
                experience = [prev_inpt_board, one_hot_reward, inpt_board]
                if max_exps > len(experiences):
                    experiences.append(experience)
                else:
                    experiences[index] = experience
                    index += 1
                    if index >= max_exps:
                        index = 0

            if env == LOCAL:
                #game.display()
                game = Game()
                game.add_random_tile()
            else:
                driver.find_element_by_class_name("restart-button").click()
                time.sleep(0.5)
            
            #print "Final Score: %d" % score

            if len(experiences) == max_exps:
                tot_score_counter += score
                game_counter += 1
                if game_counter == 64:
                    print "Average score over last %d games: %d" % (game_counter, tot_score_counter/game_counter)
                    # avg_scores.append(tot_score_counter/game_counter)
                    # data['experiences'] = experiences
                    # data['epochs_trained'] = c
                    # data['avg_scores'] = avg_scores
                    # pickle.dump(data, open(EXPERIENCES_PATH, 'wb'))
                    # print "Data saved!"
                    
                    game_counter = 0
                    tot_score_counter = 0

            score = 0
            board = get_board(env, game)
            inpt_board = flatten(normalize(board))
            prev_inpt_board = None
            continue

        if env == WEB:
            score_text = driver.find_element_by_class_name('score-container').text
            new_score = int(score_text.split("\n")[0])
        else:
            new_score = game.score

        reward = get_reward(new_score - score)
        score = new_score
        next_inpt_board = flatten(normalize(new_board))

        one_hot_reward = np.zeros((4))
        one_hot_reward[action] = reward
        experience = [inpt_board, one_hot_reward, next_inpt_board]

        if max_exps > len(experiences):
            experiences.append(experience)
        else:
            experiences[index] = experience
            index += 1
            if index >= max_exps:
                index = 0

        # Update board
        prev_action = action
        board = new_board
        prev_inpt_board = inpt_board
        inpt_board = next_inpt_board
        
        # Train the model using experience replay
        if c > 0 and c % batch_size == 0 and len(experiences) == max_exps:
            sample = random.sample(experiences, batch_size)
            next_state = np.array(map(lambda x: x[2], sample))
            rewards = np.array(map(lambda x: x[1], sample))
            cur_state = np.array(map(lambda x: x[0], sample))

            values = out.eval(feed_dict={inpt: next_state}, session=sess)

            y_batch = np.add(rewards, discount * values)
            sess.run([train_opt],feed_dict = {
                inpt: cur_state,
                y: y_batch})

            if c % (batch_size * 1000) == 0:
                print "Saving model!"
                save_path = saver.save(sess, MODEL_PATH)


def move(env, game, action_indices, board):
    for i in range (4):
        moved = False
        action = action_indices[i]
        if env == LOCAL:
            moved = game.move(action)
        else:   
            if action == 0:
                game.send_keys(Keys.ARROW_UP)
            elif action == 1:
                game.send_keys(Keys.ARROW_RIGHT)
            elif action == 2:
                game.send_keys(Keys.ARROW_DOWN)
            elif action == 3:
                game.send_keys(Keys.ARROW_LEFT)
            time.sleep(.1)
            new_board = get_board(env, game)
            moved = new_board != board

        if moved:
            if env == LOCAL:
                game.add_random_tile()
            
            new_board = get_board(env, game)
            return [action, new_board]

    return [-1, None]


# main
random.seed(12345)
sess = tf.Session()

if env == WEB:
    init()
    restart = driver.find_element_by_class_name("restart-button")
    restart.click()
    time.sleep(0.5) # allow game to load

inpt, out = network()
train(inpt, out, env, sess)

if env == WEB:
    driver.quit()