import os
from dqn.dqn_agent import Agent
from dqn.config import Config
from dqn.buffers import *
import argparse
from pre_processing.grammar import *
from env.mutation_env import *
import re





if __name__ == "__main__":
    track = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_spec', help='path to OpenAPI Specification from root directory', type=str, required=True)
    parser.add_argument('--auth_type', help='type of the authentication [cookie, apikey, account]', type=str, default='')
    parser.add_argument('--auth', help='value used for authentication', type=str, default = '')
    args = parser.parse_args()
    print(f'Beginning test with OpenAPi Specification: {args.api_spec}')
    config_defaults ={'gamma'           : 0.9,
                      'epsilon'         : 0.05,
                      'batch_size'      : 128,
                      'update_step'     : 100,
                      'episode_length'  : 10,
                      'learning_rate'   : 0.005,
                      'training_length' : 10000,
                      'priority'        : True,
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
        apikey = re.sub("{", "{\"",args.auth)
        apikey = re.sub("}", "\"}",apikey)
        apikey = re.sub(":", "\":\"",apikey)
        
        apikey = json.loads(apikey)
    elif args.auth_type == 'account':
        print(args.auth)
        account = re.sub("{", "{\"",args.auth)
        account = re.sub("}", "\"}",account)
        account = re.sub(":", "\":\"",account)
        
        logins = json.loads(account)
    if re.search('localhost:\d{4}/createdb', all_requests[0].path):
        import requests
        requests.get(all_requests[0].path)
        all_requests = all_requests[1:]
        fuzz_requests = all_requests
    
    mut_env = APIMutateEnv(action_space=23, request_seq=fuzz_requests, all_requests=all_requests, logins=logins, cookies=cookie, apikeys=apikey)#, apikey=apikey, cookies=cookie)
        
    print('OpenAPI Specification loaded.')
    print('Beginning Test...')
    
    action_space = mut_env.action_space.n
    obs_space = mut_env.observation_space.shape[0]
    all_states = []

    observation_mut = mut_env.reset()

    total_episodes = []

    total_rewards = []
    total_ep_lengths = []
    rolling_ep_len_avg = []
    total_successful_episodes = []

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

    response_codes = {}
    while episode < config.training_length:
        if episode % config.eps_per_endpoint == 0:
            if episode != 0 and current_request_idx + 1 < len(mut_env.requests):
                current_request_idx = current_request_idx + 1 #mut_env.requests.index(mut_env.current_request) + 1
                mut_env.requests_idx = current_request_idx
                mut_env.current_request = mut_env.requests[current_request_idx]
                mut_env.request_type = mut_env.current_request.type
                print(f'Moving to new operation: {mut_env.current_request.path},  {mut_env.current_request.type}')
                mut_env.parameter_idx = 0
            elif current_request_idx + 1 >= len(mut_env.requests):
                episode = config.training_length
                print('Moving to Security Oracle Test')
                done = True
                break

        episode_rewards = []
        episode_disc_rewards = 0

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
           
            mut_action = np.random.randint(0, action_space)

            next_observation_mut, reward, done, infos = mut_env.step(mut_action)
            all_states.append(observation_mut)

            total_reward = reward
           
            ep_response.append([infos['action'], infos['status'], infos['method'], infos['request']])
            statuses.append(infos['status'])
            episode_rewards.append(reward)
            observation_mut = next_observation_mut
    
            if infos['status'] in response_codes.keys():
                response_codes[infos['status']] += 1
            else:
                response_codes[infos['status']] = 1


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
        total_successful_episodes.append(len(np.where(np.mean(episode_rewards) == 0)[0]))

        total_episodes.append(episode)
        total_rewards.append(sum(episode_rewards))
        total_ep_lengths.append(step)
        episode += 1

    mut_env.close()
    two_hundred_requests = sum([response_codes[key] for key in response_codes.keys() if str(key)[0] == '2'])
    five_hundred_requests = sum([response_codes[key] for key in response_codes.keys() if str(key)[0] == '5'])
    apirl_loc_uniq = set()
    for req in mut_env.error_endpoints:
        apirl_loc_uniq.add(str((req[0].path, req[5].request.method)))
    
    save_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Saving testing info to ./logs/{save_time}_random/")
    path = os.path.abspath(os.getcwd())
    if not os.path.exists(path + '/logs/'):
        os.mkdir(path +'/logs')
    os.mkdir(path + f'./logs/{save_time}_random/')

    with open(f'./logs/{save_time}_random/error_endpoints.pkl', 'wb') as f:
        pkl.dump(apirl_loc_uniq, f)
    
    with open(f'./logs/{save_time}_random/request_response.yaml', 'w') as yaml_out:
        yaml.dump(response, yaml_out)


    print(f"200+500%: {(two_hundred_requests+five_hundred_requests)/sum([value for value in response_codes.values()])}")
    print(f"200%: {two_hundred_requests/sum([value for value in response_codes.values()])}")
    print(f"500s: {five_hundred_requests}")
    print(f"reqs: {sum([value for value in response_codes.values()])}")
    print(f"Unique endpoints with 500s: {len(apirl_loc_uniq)}")
    for endpoint in apirl_loc_uniq:
        print(endpoint)




