#!/usr/bin/env python

import tensorflow as tf
import cv2
import sys
import datetime
sys.path.append("Wrapped Game Code/")
import pong_fun_touch as game# whichever is imported "as game" will be used
import tetris_fun
import random
import numpy as np
import os
from collections import deque

GAME = 'pong' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 500. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 100000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

######################################################################
start = datetime.datetime.now()
store_network_path="temp/my_networks/"
tensorboard_path = "temp/logs/"
out_put_path = "temp/logs_" + GAME

if os.path.exists('temp'):
    pass
else:
    os.makedirs(store_network_path)
    os.makedirs(out_put_path)

pretrain_number=0


######################################################################
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    Q_max=tf.reduce_max(readout)
    tf.scalar_summary('Q_max', Q_max)

    return s, readout

def sencond2time(senconds):

	if type(senconds)==type(1):
		h=senconds/3600
		sUp_h=senconds-3600*h
		m=sUp_h/60
		sUp_m=sUp_h-60*m
		s=sUp_m
		return ",".join(map(str,(h,m,s)))
	else:
		return "[InModuleError]:sencond2time(senconds) invalid argument type"

def trainNetwork(s,s_hit, readout,readout_hit, sess,sess2):

    
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    
    # cost_network
    a_hit = tf.placeholder("float", [None, ACTIONS])
    y_hit = tf.placeholder("float", [None])
    readout_action_hit = tf.reduce_sum(tf.mul(readout_hit, a_hit), reduction_indices = 1)
    cost_hit = tf.reduce_mean(tf.square(y_hit - readout_action_hit))
    train_step_hit = tf.train.AdamOptimizer(1e-6).minimize(cost_hit)
    

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open(out_put_path  + "/readout.txt", 'w')
    h_file = open(out_put_path  + "/hidden.txt", 'w')


    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    sess2.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state(store_network_path)
    
    #saver.restore(sess, "new_networks/pong-dqn-"+str(pretrain_number))    
    
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
	#saver.restore(sess, "my_networks/pong-dqn-26000")
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"
    
    print "Press any key and Enter to continue:"
    raw_input()

    epsilon = INITIAL_EPSILON
    t = 0
    total_score=0
    positive_score=0
    
    Average_counter=0
    Average_number1=0
    Average_number2=0
    while True:
        # choose an action epsilon greedily
        readout_t = sess.run(readout,feed_dict = {s : [s_t]})[0]
        hit_t  = sess2.run(readout_hit,feed_dict = {s_hit : [s_t]})[0]
        

        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            
            action_index_1 = np.argmax(readout_t)           
            action_index_2 = np.argmax(hit_t)            
            
            number_1=readout_t[action_index_1]
            number_2=hit_t[action_index_2]            
            
            if Average_counter==0:
                Average_number1=0
                Average_number2=0
            else:
                Average_number1= Average_number1 + (number_1-Average_number1)/t 
                Average_number2= Average_number2 + (number_2-Average_number1)/t
                
            Average_counter+=1
            
            
            
            if (number_1/Average_number1)>(number_2/Average_number2):
                a_t[action_index_1] = 1
            else:
                a_t[action_index_2] = 1
#            action_index = np.argmax(readout_t)
#            a_t[action_index] = 1
            
        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K):
            # run the selected action and observe next state and reward
            x_t1_col, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
        if r_t==-1 or r_t==1:
            total_score=total_score+r_t;
            
        if r_t==1:
            positive_score=positive_score+r_t

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            y_batch_hit = []
            r_batch_temp=np.zeros(32)
            
            readout_j1_batch = sess.run(readout,feed_dict = {s : s_j1_batch})
            readout_hit_j1_batch = sess2.run(readout_hit,feed_dict = {s_hit : s_j1_batch})
            
            for i in range(0, len(minibatch)):
                
                
                #change reward function
                if r_batch[i]==0.5:
                    r_batch_temp[i]=0
                elif r_batch[i]==1:
                    r_batch_temp[i]=1
                elif r_batch[i]==-1:
                    r_batch_temp[i]=-1
                else:
                    r_batch_temp[i]=0
                
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch_temp[i])
                else:
                    #y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    y_batch.append(r_batch_temp[i] + GAMMA * np.max(readout_j1_batch[i]))   
                    
                    
            
            for i in range(0, len(minibatch)):
                
                
                #change reward function
                if r_batch[i]==0.5:
                    r_batch_temp[i]=1
                elif r_batch[i]==1:
                    r_batch_temp[i]=0
                elif r_batch[i]==-1:
                    r_batch_temp[i]=-1
                else:
                    r_batch_temp[i]=0
                
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch_hit.append(r_batch_temp[i])
                else:
                    #y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    y_batch_hit.append(r_batch_temp[i] + GAMMA * np.max(readout_hit_j1_batch[i]))  
                    

            # perform gradient step
            sess.run(train_step,feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch})
            
            # perform gradient step
            sess2.run(train_step_hit,feed_dict = {
                y_hit : y_batch_hit,
                a_hit : a_batch,
                s_hit : s_j_batch})
            

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, store_network_path + GAME + '-dqn', global_step = t+pretrain_number)
            
            #saver.save(sess, 'new_networks/' + GAME + '-dqn', global_step = t)

        if t % 500 == 0:  
            now=datetime.datetime.now()
            diff_seconds=(now-start).seconds
            time_text=sencond2time(diff_seconds)
            
#            result = sess.run(merged,feed_dict = {s : [s_t]})
#            writer.add_summary(result, t+pretrain_number)
            a_file.write(str(t+pretrain_number)+','+",".join([str(x) for x in readout_t]) + \
            ','+str(total_score)+ ','+str(positive_score) \
            +','+time_text+','+",".join([str(x) for x in hit_t])+','+str(Average_number1)+','+str(Average_number2)+'\n')

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print "TIMESTEP:", t+pretrain_number, "/ ACTION:", action_index, "/ REWARD:", r_t, "/ Q_MAX: %e" % np.max(readout_t),'  time:(H,M,S):' \
        + sencond2time((datetime.datetime.now()-start).seconds)
        print 'Total score:',total_score,' Positive_score:',positive_score,'   up:',readout_t[0],'    down:',readout_t[1],'  no:',readout_t[2]
       
        # write info to files
        
        #if t % 10000 <= 100:
            #a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            #h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            #cv2.imwrite("logs_pong/frame" + str(t) + ".png", x_t1)
        

def playGame():
    sess = tf.Session()
    sess2 =tf.Session()

    s, readout = createNetwork()
    s_hit,readout_hit= createNetwork()

#    merged = tf.merge_all_summaries()
#    writer = tf.train.SummaryWriter(tensorboard_path, sess.graph)
    
    trainNetwork(s,s_hit, readout,readout_hit, sess,sess2)

def main():
    playGame()

if __name__ == "__main__":
    main()
