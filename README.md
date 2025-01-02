```
 ________   ______   ________  ______    __          
/_______/\ /_____/\ /_______/\/_____/\  /_/\         
\::: _  \ \\:::_ \ \\__.::._\/\:::_ \ \ \:\ \        
 \::(_)  \ \\:(_) \ \  \::\ \  \:(_) ) )_\:\ \       
  \:: __  \ \\: ___\/  _\::\ \__\: __ `\ \\:\ \____  
   \:.\ \  \ \\ \ \   /__\::\__/\\ \ `\ \ \\:\/___/\ 
    \__\/\__\/ \_\/   \________\/ \_\/ \_\/ \_____\/
```

APIRl is a Deep Reinforcement Learning tool to find bugs in REST APIs. The paper detailing APIRL can be found online on [arXiv](https://arxiv.org/abs/2412.15991) or AAAI 2025 (coming soon).



## Installation

APIRL runs on Python 3.9. We provide two ways of creating the python environment to run APIRL:
- Conda Environment: `conda env create -f apirl_environment.yml`
- pip requirements: `pip install -r requirements.txt`


## Quickstart

We provide two saved models to quickly start running APIRL on new APIs in the `saved_models` directory for APIRL and APIRL-cov (a version of APIRL trained on coverage feedback).
You can run APIRL in a number of different ways including:
```
python run_apirl.py --api_spec /openapi_specs/example.json --auth_type apikey --auth {'Authorization': api_key}
```
### OpenAPI Specification
APIRL requires an openAPI specification as input to begin trying to find bugs in REST APIs. This file should specify the URIs for endpoints and the domain (including any ports).

### Authentication
If not provided with an authentication mechanism via the command line flags APIRL will try to authenticate with the API if possible using the `self.generate_token()` function in `env/mutation_env.py`.

Alternatively, APIRL can be authenticated with an API using a number of different mechanisms and can be specified with the `--auth_type` flag in the following ways:
- `--auth_type cookie --auth {'header value': cookie}`
- `--auth_type apikey --auth {'header value': api_key}`
- `--auth_type account --auth {'username': 'password'}`

These do not assume to have a timeout on any of the authentication mechanisms.

### Specifying APIRL types
APIRL by default loads the models saved in the `saved_models/dqn` directory. You may specify which model to load by using the `--model` flag.

We include another version of APIRL (APIRL-cov) that can be loaded by `--model saved/models/dqn_coverage/dqn.pt`

### Environments

We provide implementations of the different environments used in the ablations of APIRL. When running the `run_april.py` file, the additional argument `--env`  can be used to set a number of ablations:
 
- APIRL-r 		: `--env ratio`
- APIRL-u 		: `--env binary`
- APIRL-m 		: `--env no-transformer`
- APIRL-arat 	: `--env aratrl`	


## Training

APIRL can be trained using the python script `train_apirl.py` using the `--api_spec`, `--auth_type`, `--auth`, and `--env` flags. Our version of APIRL is trained using [Generic University](https://github.com/InsiderPhD/Generic-University), details to set up the API can be found in the repo.

### Transformer training 

The APIRL transformer can be trained via the python script `pre_processing/train_roberta.py`, it should be run from the `pre_processing/` directory. By default it can be trained using the dataset we provide, or if you wish to train on your own dataset you can change the path in on line 93 of `train_roberta.py`.

## API Dataset

The REST API dataset used to train the APIRL transformer for internal representation can be found in the file `pre_processing/api_dataset.txt`/



## Citation
If you use our code, dataset, or check out the paper, please make sure to cite us!
```
@inproceedings{foley_apirl_2025,
	title = {{APIRL}: {Deep} {Reinforcement} {Learning} for {REST} {API} {Fuzzing}},
	copyright = {Copyright (c) 2025 Association for the Advancement of Artificial Intelligence},
	shorttitle = {{ALPHAPROG}},
	booktitle = {Proceedings of the {AAAI} {Conference} on {Artificial} {Intelligence}},
	author = {Foley, Myles and Maffeis, Sergio},
	year = {2025},
}

```