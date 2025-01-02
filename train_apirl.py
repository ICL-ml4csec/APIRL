#import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import random
import time
from matplotlib import pyplot as plt
import numpy as np
from collections import deque
from dqn.dqn_agent import Agent
from dqn.config import Config
import torch
import os
import inspect
from dqn.buffers import *
from env.mutate_env_ma import *
#from stable_baselines3 import DQN
from env.request_seq_env import *
from env.auth_env import *
from pre_processing.grammar import *
import yaml
from rnd import *
from env.enumerate_api import MakeGraph
from env.trigger_ac_env import APITriggerEnv



if __name__ == "__main__":
    track = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_spec', help='path to OpenAPI Specification from root directory', type=str, required=True)
    parser.add_argument('--auth_type', help='type of the authentication [cookie, apikey, account]', type=str, default='')
    parser.add_argument('--auth', help='value used for authentication', type=str, default='')
    parser.add_argument('--env', help='used to specify ablations of reward or transformer', type=str, default='default')  


    args = parser.parse_args()
    print(f'Beginning test with OpenAPi Specification: {args.api_spec}')

    config_defaults ={'gamma'           : 0.9,
                      'epsilon'         : 1.0,
                      'batch_size'      : 128,
                      'update_step'     : 500,
                      'episode_length'  : 10,
                      'learning_rate'   : 0.005,
                      'training_length' : 10000,
                      'priority'        : True,
                      'exploring_steps' : 128,
                      'rnd'             : False,
                      'pre_obs_norm'    : 10,
                      'attention'       : True,
                      'eps_per_endpoint': 3}
    config = Config(config_dict=config_defaults)


    show_train_curve = not track

    spec_path = os.path.abspath(os.getcwd()) + args.api_spec
    try:
        all_requests = create_requests(spec_path)
    except:
        all_requests = custom_parse(spec_path)
    all_requests = remove_delete_requests(all_requests)
    fuzz_requests = all_requests
    logins = cookie = apikey = None
    if args.auth_type == 'cookie':
        cookie = {'Cookie': args.auth}
    elif args.auth_type == 'apikey':
        apikey = json.loads(args.auth.replace('\'' ,'"'))
    elif args.auth_type == 'account':
        logins = json.loads(args.auth.replace('\'' ,'"'))
    #apikey = {'Authorization-Token': 'YWRtaW46cGFzczE='} # used for vapi
    if re.search('localhost:\d{4}/createdb', all_requests[0].path):
        import requests
        requests.get(all_requests[0].path)
        all_requests = all_requests[1:]
        fuzz_requests = all_requests
    
    if args.env =='default':
        from env.mutation_env import *
    elif args.env =='aratrl':
        from env.ablation_envs.aratrl_reward import *
    elif args.env =='binary':
        from env.ablation_envs.binary_reward import *
    elif args.env =='no-transformer':
        from env.ablation_envs.simple_state import *
    elif args.env =='ratio':
        from env.ablation_envs.ratio_reward import *
    else:
        print('Environment not does not exist!')
        print('Please choose from: [default, aratrl, binary, no-transformer, ratio]')
        exit(-1)

    mut_env = APIMutateEnv(action_space=23, request_seq=fuzz_requests, all_requests=all_requests, logins=logins)
    print('Environment Initalised...')

    action_space = mut_env.action_space.n
    obs_space = mut_env.observation_space.shape[0]
    mut_agent = Agent(action_space=action_space, state_space=obs_space, gamma=config.gamma, rnd=False,
                      epsilon=config.epsilon, lr=config.learning_rate, batch_size=config.batch_size, model='dqn')

    print('Agent Created...')
    observation_mut = mut_env.reset()

    total_episodes = []
    total_losses = []
    total_rewards = []
    total_ep_lengths = []
    rolling_ep_len_avg = []
    total_successful_episodes = []
    mut_xp_buffer = PriorityReplayBuffer() if config.priority else ReplayBuffer()
    print('filling buffer...')
    while config.batch_size > len(mut_xp_buffer):
        observation_mut = mut_env.reset()
        for step in range(0, config.episode_length):
            req_action = np.random.randint(0, action_space)
            next_observation_mut, reward, done, infos = mut_env.step(req_action)
            mut_xp_buffer.add_transition([observation_mut, req_action, reward, next_observation_mut, done])
            observation_mut = next_observation_mut
    print('Training...')

    total_step_number = 0
    num_eps_this_form = 0

    episode = 0
    start_time = time.time()
    number_of_error_requests = [0]
    number_of_400_requests = [0]
    number_of_401_requests = [0]
    number_of_403_requests = [0]
    number_of_404_requests = [0]
    number_of_405_requests = [0]
    number_of_200_requests = [0]
    number_of_201_requests = [0]
    number_of_204_requests = [0]
    number_of_40X_requests = [0]
    number_of_20X_requests = [0]
    number_of_X0X_requests = [0]
    current_request_idx = 0
    response = {}

    action_dist = {a:0 for a in range(action_space)}

    while episode < config.training_length:
        if episode % 1000 == 0:
            if episode != 0 and current_request_idx + 1 < len(mut_env.requests):
                current_request_idx = current_request_idx + 1
                mut_env.requests_idx = current_request_idx
                mut_env.current_request = mut_env.requests[current_request_idx]
                mut_env.request_type = mut_env.current_request.type
                mut_agent.epsilon = 0.6
                print('moving to new endpoint:')
                print(mut_env.current_request.path + ' ' + mut_env.current_request.type)
                mut_env.parameter_idx = 0
            elif current_request_idx + 1 >= len(mut_env.requests):
                episode = config.training_length
                print('done')
                done = True
                break
        ep_loss = []
        episode_rewards = []
        episode_disc_rewards = 0
        #observations = req_env.reset()
        observations = mut_env.reset()
        statuses = []
        done = False
        ep_response = []
        step = 0
        ep_400_requests = 0
        ep_401_requests = 0
        ep_403_requests = 0
        ep_404_requests = 0
        ep_405_requests = 0
        ep_200_requests = 0
        ep_204_requests = 0
        ep_500_requests = 0

        ep_201_requests = 0
        ep_20X_requests = 0
        ep_40X_requests = 0
        ep_X0X_requests = 0

        while step < config.episode_length and done == False:
            mut_minibatch = mut_xp_buffer.sample(config.batch_size)
            mut_action = mut_agent.get_action(observation_mut)
            action_dist[mut_action] = action_dist[mut_action] + 1
            next_observation_mut, reward, done, infos = mut_env.step(mut_action)
            if config.rnd:
                intrinsic_reward = mut_agent.rnd.compute_intrinsic_reward(next_observation_mut)
                reward += intrinsic_reward.clamp(-1.0, 1.0).item()
            
            total_reward = reward
            loss = mut_agent.dqn.train_q_network(mut_minibatch, rnd=mut_agent.rnd, priority=config.priority)
            if config.rnd:
                mut_agent.rnd.update(mut_minibatch)

            mut_xp_buffer.add_transition([observation_mut, mut_action, reward, next_observation_mut, done])
            if config.priority:
                loss, priorities = loss
                mut_xp_buffer.update_priorities(mut_minibatch[4], priorities)
            ep_response.append([infos['action'], infos['status'], infos['method'], infos['request']])
            statuses.append(infos['status'])
            episode_rewards.append(reward)
            observation_mut = next_observation_mut

            ep_loss.append(loss)
            if infos['status'] == 400:
                ep_400_requests += 1
            if infos['status'] == 401:
                ep_401_requests += 1
            if infos['status'] == 403:
                ep_403_requests += 1
            elif infos['status'] == 404:
                ep_404_requests += 1
            elif infos['status'] == 200:
                ep_200_requests += 1
            elif infos['status'] == 201:
                ep_201_requests += 1
            elif infos['status'] == 204:
                ep_204_requests += 1
            elif infos['status'] == 405:
                ep_405_requests += 1
            elif infos['status'] == 500:
                ep_500_requests += 1

            if 200 <= infos['status'] <= 299:
                ep_20X_requests += 1
            elif 400 <= infos['status'] <= 499:
                ep_40X_requests += 1
            else:
                ep_X0X_requests += 1
            step += 1

        response[episode] = ep_response
        mut_agent.update_epsilon()

        if episode % config.update_step == 0 and episode != 0:
            print('Updating Target Network...')
            mut_agent.update_network()


        number_of_error_requests.append(number_of_error_requests[episode]+ep_500_requests)
        number_of_400_requests.append(number_of_400_requests[episode]+ep_400_requests)
        number_of_401_requests.append(number_of_401_requests[episode]+ep_401_requests)
        number_of_403_requests.append(number_of_403_requests[episode]+ep_403_requests)
        number_of_404_requests.append(number_of_404_requests[episode]+ep_404_requests)
        number_of_405_requests.append(number_of_405_requests[episode]+ep_405_requests)

        number_of_201_requests.append(number_of_201_requests[episode]+ep_201_requests)
        number_of_200_requests.append(number_of_200_requests[episode]+ep_200_requests)
        number_of_204_requests.append(number_of_204_requests[episode]+ep_204_requests)

        number_of_20X_requests.append(number_of_20X_requests[episode]+ep_20X_requests)
        number_of_40X_requests.append(number_of_40X_requests[episode]+ep_40X_requests)
        number_of_X0X_requests.append(number_of_X0X_requests[episode]+ep_X0X_requests)

        num_eps_this_form += 1


        if 1:
            print("{:<5}{:<6}{:>2}{:<15}{:>.3f}{:<15}{:>.3f}{:<22}{:>.3f} {:<.3f} {:> .3f} {:<.3f}{:<40}".format(
                    str(episode),
                    'AGENT: ', 1,
                    ' EP_LOSS_AV: ', float(np.mean(ep_loss)/(step+1)) if ep_loss else 0,
                    ' EP_REWARD: ', float(sum(episode_rewards)),
                    ' REWARD MIN/MAX/MEAN/SD: ', float(min([list(episode_rewards)[j] for j in range(len(episode_rewards))])),
                    float(max([list(episode_rewards)[j] for j in range(len(episode_rewards))])), float(np.mean([float(list(episode_rewards)[j]) for j in range(len(episode_rewards))])),
                    float( np.std([list(episode_rewards)[j] for j in range(len(episode_rewards))])),
                               ' ACTION: ' + str(infos['action'])
            ))
            print(statuses)
            total_successful_episodes.append(len(np.where(np.mean(episode_rewards) == 0)[0]))
            total_losses.append(sum(ep_loss)/step if ep_loss else 0)
            total_episodes.append(episode)
            total_rewards.append(sum(episode_rewards))
            total_ep_lengths.append(step)
        episode += 1
    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Total run time: ')
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


    mut_env.close()
    dir_out = mut_agent.save_model()
    print('saved to: '+ dir_out)
    with open('./' + dir_out + '/request_response.yaml', 'w') as yaml_out:
        yaml.dump(response, yaml_out)

    if show_train_curve:
        print('number of 200 requests: ' + str(number_of_200_requests[-1]))
        print('number of 201 requests: ' + str(number_of_201_requests[-1]))
        print('number of 204 requests: ' + str(number_of_204_requests[-1]))
        print('number of 400 requests: ' + str(number_of_400_requests[-1]))
        print('number of 401 requests: ' + str(number_of_401_requests[-1]))
        print('number of 403 requests: ' + str(number_of_403_requests[-1]))
        print('number of 404 requests: ' + str(number_of_404_requests[-1]))
        print('number of 405 requests: ' + str(number_of_405_requests[-1]))
        print('number of 500 requests: ' + str(number_of_error_requests[-1]))

        plt.plot(total_episodes, total_rewards, color='orange')
        plt.title('Mean reward of all agents in the episode')
        plt.grid()
        plt.savefig('./' + dir_out + '/reward.png')
        plt.show()

        plt.plot(total_episodes, total_losses)
        plt.title('Mean Value loss of all agents in the episode')
        plt.grid()
        plt.xlabel('Episode')
        plt.ylabel('MSE of Advantage')
        plt.savefig('./' + dir_out + '/loss.png')
        plt.show()

        import pickle as pkl
        with open( f'./{dir_out}/loss.pkl', 'wb') as f:
            pkl.dump(total_losses, f)

        plt.plot(total_episodes, number_of_200_requests[1:])
        plt.plot(total_episodes, number_of_201_requests[1:])
        plt.plot(total_episodes, number_of_204_requests[1:])
        plt.plot(total_episodes, number_of_400_requests[1:])
        plt.plot(total_episodes, number_of_401_requests[1:])
        plt.plot(total_episodes, number_of_403_requests[1:])
        plt.plot(total_episodes, number_of_404_requests[1:])
        plt.plot(total_episodes, number_of_405_requests[1:])
        plt.plot(total_episodes, number_of_error_requests[1:])
        plt.title('Test Cases')
        plt.grid()
        plt.xlabel('Episode')
        plt.ylabel('Number of requests')
        plt.legend(['200', '201', '204','400', '401', '403', '404', '405', '500'])
        plt.savefig('./' + dir_out + '/40x.eps', format='eps')
        plt.show()
        with open( f'./{dir_out}/requests_verbose.pkl', 'wb') as f:
            pkl.dump({'200': number_of_200_requests,
                      '201': number_of_201_requests,
                      '204': number_of_204_requests,
                      '400': number_of_400_requests,
                      '401': number_of_401_requests,
                      '404': number_of_404_requests,
                      '405': number_of_405_requests,
                      'err': number_of_error_requests}, f)

        plt.plot(total_episodes, number_of_20X_requests[1:])
        plt.plot(total_episodes, number_of_40X_requests[1:])
        plt.plot(total_episodes, number_of_X0X_requests[1:])
        plt.title('Test Cases')
        plt.grid()
        plt.xlabel('Episode')
        plt.ylabel('Number of requests')
        plt.legend(['20X', '40X', 'X0X'])
        plt.savefig('./' + dir_out + '/statuses.eps', format='eps')
        plt.show()

        with open( f'./{dir_out}/requests.pkl', 'wb') as f:
            pkl.dump({'20X': number_of_20X_requests,
                      '40X': number_of_404_requests,
                      'XOX': number_of_X0X_requests}, f)
        plt.plot(total_episodes, number_of_error_requests[1:])
        plt.title('Number of non-unique bugs found')
        plt.grid()
        plt.xlabel('Episode')
        plt.ylabel('Bugs')
        plt.savefig('./' + dir_out + '/bugs.png')
        plt.show()

        plt.bar(action_dist.keys(), action_dist.values(), 1, color='g')
        plt.title('Action distribution')
        plt.grid()
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.savefig('./' + dir_out + '/actions.eps', format='eps')
        plt.show()








