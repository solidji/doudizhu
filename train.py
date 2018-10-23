# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import absolute_import
from game.agent import Agent
import numpy as np
from game.rlutil import combine
from rl.init_model import model_init
from game.config import Config
#rl
if __name__=="__main__":
    #J 20000与随机训练胜率76%-90%(修改网络模型前)
    #J 在虚拟机上跑，0-100000 rand,rand,花费50分钟,跑完之后win_rate:  [0.32967033 0.33366633 0.33666334] [330. 334. 337.]
    #J episode:  100000 , epsilon:  0.99 , loss:  3.6692226 , win_rate:  [0.12756872 0.43801562 0.43441566] [12757. 43802. 43442.] 0.7073629263707363 2
    #J 10W-20W与cxgz，cxgz训练
    #J episode:  200000 , epsilon:  0.99 , loss:  4.0304704 , win_rate:  [0.02617974 0.499525   0.47429526] [ 2618. 49953. 47430.] 0.5759842401575984 1
    #J episode:  400000 , epsilon:  0.99 , loss:  2.7135177 , win_rate:  [0.05687472 0.48627757 0.45684772] [11375. 97256. 91370.] 0.5940970295148524 2
    step = 0
    num_epochs = 800001
    rl_model = "prioritized_dqn"
    start_iter=600000
    
    my_config = Config()
    learning_rate = 0.0001
    e_greedy = 0.99
    
    RL = model_init(my_config, rl_model, e_greedy=e_greedy, start_iter=start_iter, epsilon_init=0.85,  e_greedy_increment=0.00001)
    agent = Agent(models=["rl","cxgz","cxgz"], my_config=my_config, RL=RL, train=False)
    
    losss = []
    winrates = []
    es = []
    
    f = open('log.txt', 'a')
    winners = np.zeros(3)
    win_rate = winner_1 = 0
    learn_step_counter = 0
    loss = 0
    for episode in range(start_iter, num_epochs):
        # initial observation
        s = agent.reset() #返回当前player的手牌，目前写死是player1
        if episode%5000 == 0:
            print(agent.game.playrecords.show("==================="+str(episode)+"==================="))
        done = False
        
        first = True
        start = 0
        while(not done):
            
            #随机开局
            if first:
                # start = np.random.randint(0,3)
                agent.game.i = agent.game.dizhu
                
            # RL choose action based on observation
            actions = agent.get_actions_space() #返回可可能的出牌组合
            s = combine(s, actions)
            #action to one-hot
            actions_one_hot = np.zeros(agent.dim_actions)
            for k in range(len(actions)):
                actions_one_hot[actions[k]] = 1
                
            action, action_id, q, q_oh, q_av = RL.choose_action(s, actions_one_hot, actions)
            
            # RL take action and get next observation and reward
            s_, r, done = agent.step(action_id=action_id)
            
            actions_ = agent.get_actions_space_state()
            #action to one-hot
            actions_one_hot_ = np.zeros(agent.dim_actions)
            for k in range(len(actions_)):
                actions_one_hot_[actions_[k]] = 1
                
            s_ = combine(s_, actions_) #get_actions_space_state不改变game参数
            
            if not first or start == 0:
                RL.store_transition(s, actions_one_hot_, action, r, s_)

            if (step > 500) and (step % 15 == 0):
                loss, learn_step_counter = RL.learn()
                em_name, em_value, e_name,e_value, t_name, t_value = RL.check_params()

            # swap observation
            s = s_

            step += 1
            
            first = False
        
        if agent.game.playrecords.winner == 1:
            winners[0] = winners[0] + 1
        elif agent.game.playrecords.winner == 2:
            winners[1] = winners[1] + 1
        elif agent.game.playrecords.winner == 3:
            winners[2] = winners[2] + 1


        if agent.game.playrecords.winner == 1 or agent.game.playrecords.winner != agent.game.dizhu + 1:
            winner_1 = winner_1 + 1
        win_rate_1 = winner_1/np.sum(winners)
        
        win_rate = winners/np.sum(winners)
        #print(agent.game.get_record().records)
        #print(r)
        e = RL.epsilon
        if episode%1000 == 0:
            losss.append(loss)
            winrates.append(win_rate[0])
            es.append(e)
        #    winners = np.zeros(3)
            
        if episode%1000 == 0:
            #保存模型
            if episode%50000 == 0 and episode != start_iter:
                model ="Model_dqn/"+str(episode)+".ckpt"
                RL.save_model(model)
                print("save: ",episode)
            print("episode: ",episode,", epsilon: ", e, ", loss: ", loss, ", win_rate: ",win_rate, winners, win_rate_1, agent.game.dizhu)
            print(agent.game.get_record().records)
            
            f.write("episode: "+ str(episode) + ", epsilon: "+ str(e) + ", loss: "+ str(loss) + ", win_rate: "+ str(win_rate))
            f.write("\n")
            #f.write(str(agent.game.get_record().records))
            #f.write("\n")
            f.flush()
            
    # end of game
    print('game over')
    f.close()
    RL.plot_cost()
    