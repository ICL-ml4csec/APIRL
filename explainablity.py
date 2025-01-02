import shap
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from dqn.dqn_agent import Agent
from dqn.config import Config
import numpy as np
import os.path as path


if __name__ == '__main__':

    config_defaults = {'gamma'           : 0.9,
                          'epsilon'         : 0.05,
                          'batch_size'      : 128,
                          'update_step'     : 100,
                          'episode_length'  : 10,
                          'learning_rate'   : 0.005,
                          'training_length' : 10000,
                          'priority'        : True,
                          'eps_per_endpoint': 3}

    config = Config(config_dict=config_defaults)
    action_space = 23
    obs_space = 772
    mut_agent = Agent(action_space=action_space, state_space=obs_space, gamma=config.gamma,
                      epsilon=config.epsilon, lr=config.learning_rate, batch_size=config.batch_size, model='dqn')

    one_up = path.abspath(path.join(__file__, "../../"))

    mut_agent.load_model(one_up + '/APIRL/saved_models/dqn/dqn.pt')


    with open(one_up + '/experiment_results/explainablity/states.pkl', 'rb') as f:
        mutate_states = pkl.load(f)
    mutate_states = np.array(mutate_states)
    param_type_feature = []
    http_type = []
    http_response_code = []
    param_location = []
    auto_features = dict()
    auto_features_raw = []
    for i in range(mutate_states.shape[0]):
        param_type_feature.append(mutate_states[i,0])
        http_type.append(mutate_states[i,1])
        http_response_code.append(mutate_states[i,2])
        param_location.append(mutate_states[i,3])

        if len(auto_features) == 0 or not np.any(
                np.all(mutate_states[i, 4:] == np.array(list(auto_features.values())), axis=1)):
                auto_features[len(auto_features)] = mutate_states[i, 4:]
                auto_features_raw.append(len(auto_features)-1)
        else:
            for key, arr in auto_features.items():
                if np.array_equal(arr, mutate_states[i, 4:]):
                    auto_features_raw.append(key)
    mut_agent.auto_features = auto_features
    mutate_features = pd.DataFrame({'Parameter Type': param_type_feature, 'Parameter Location': param_location,
                                     'HTTP Method': http_type, 'HTTP Response Code': http_response_code, 'Embedded Features': auto_features_raw})
    data = 'a'
    explainer = shap.KernelExplainer(model=mut_agent.get_action_shap, data=mutate_features.head(1))

    m_shap_values = explainer.shap_values(mutate_features)
    shap.summary_plot(m_shap_values[0], mutate_features, show=False)
    plt.gcf().axes[-1].set_aspect(20)
    plt.gcf().axes[-1].set_box_aspect(20)
    plt.gca.legend_=None
    from datetime import datetime

    dt = datetime.now()

    plt.savefig(one_up + f'/experiment_results/explainablity/dqn_{dt.strftime("%d-%m-%Y_%H-%M-%S")}_shap.pdf', format='pdf', legend=False)
    plt.show()



