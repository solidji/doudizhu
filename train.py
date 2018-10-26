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
    num_epochs = 1100001
    rl_model = "prioritized_dqn"
    start_iter = 1000000

    
    my_config = Config()
    learning_rate = 0.001
    e_greedy = 0.8
    
    RL = model_init(my_config, rl_model, e_greedy=e_greedy, start_iter=start_iter, epsilon_init=0.8,  e_greedy_increment=0.00001)
    agent = Agent(models=["rl","self","self"], my_config=my_config, RL=RL, train=False)
    
    losss = []
    winrates = []
    es = []
    
    f = open('log.txt', 'a')
    winners = np.zeros(5) # 1 2 3 1or队友 1+2+3
    winners_last = np.zeros(5)
    # winners_recent = np.zeros(4)
    # recent_rate = np.zeros(3)
    # win1_rate = winner1 = winner_recent = winner1_last = recent_rate1 = 0
    learn_step_counter = 0
    loss = 0
    for episode in range(start_iter, num_epochs):
        # initial observation
        s = agent.reset() #返回当前player的手牌，目前写死是player1

        # if episode%2000 == 0:
        #     print(agent.game.playrecords.show("==================="+str(episode)+"==================="))

        done = False
        
        first = True
        start = 0
        while(not done):
            
            #随机开局
            if first:
                # start = np.random.randint(0,3)
                agent.game.i = agent.game.dizhu
                
            # RL choose action based on observation
            actions = agent.get_actions_space() #返回可能的出牌组合
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

            if (step > 500) and (step % 50 == 0):
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
        winners[4] = winners[4] + 1

        if agent.game.playrecords.winner == 1:
            winners[3] = winners[3] + 1
        elif agent.game.playrecords.winner != agent.game.dizhu + 1 \
                and agent.game.dizhu != 0:  # 胜的是另外一个农民
            winners[3] = winners[3] + 1

        # win1_rate = winner1/np.sum(winners)
        win_rate = winners/winners[4]
        #print(agent.game.get_record().records)
        #print(r)
        e = RL.epsilon

        if episode % 2000 == 0:
            winners_recent = winners - winners_last # 最近一批2000次各玩家的获胜次数
            winners_last = winners.copy()
            # winner_recent = winner1 - winner1_last # 最近一批次玩家1以及队友的获胜次数
            # winner1_last = winner1
            # recent_rate = winners_recent / np.sum(winners_recent) # 最近一批次各自胜率
            # recent_rate1 = winner_recent / np.sum(winners) # 最近一批次玩家1以及队友胜率

            losss.append(loss)
            winrates.append(winners/winners[4])
            es.append(e)
        #   winners = np.zeros(3)

        if episode%2000 == 0:
            #保存模型
            if episode%50000 == 0 and episode != start_iter:
                model ="Model_dqn/"+str(episode)+".ckpt"
                RL.save_model(model)
                print("save: ",episode)
            print("episode: ", episode,", epsilon: ", '%.2f' % e, ", loss: ", '%.2f' % loss,
                  ", win_rate: ", winners, np.round((winners[:4]/winners[4]), decimals=4),
                  ", recent_rate: ", winners_recent, np.round((winners_recent[:4]/winners_recent[4]), decimals=4))
            # print(agent.game.get_record().records)
            
            f.write("episode: "+ str(episode) + ", epsilon: "+ str(e) + ", loss: "+ str(loss) +
                    ", win_rate: "+ str(winners_recent[:4]/winners_recent[4]))
            f.write("\n")
            #f.write(str(agent.game.get_record().records))
            #f.write("\n")
            f.flush()
            
    # end of game
    print('game over')
    f.close()
    RL.plot_cost()
    