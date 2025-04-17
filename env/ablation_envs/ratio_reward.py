import html
import time
from copy import deepcopy, copy
from requests import Response
import gym
from gym import spaces
from gym.utils import seeding
import itertools
import re
from functools import reduce
from bs4 import BeautifulSoup
import numpy as np
import urllib
import json
import random
import pandas as pd
from datetime import datetime
import string
from env.dynamic_data_storage import DynamicStorage
from env.api_interface import Interface
import html
import statistics
import requests
import os.path as path
from transformers import pipeline, PreTrainedTokenizerFast
from tqdm import tqdm


class APIMutateEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, action_space, request_seq, all_requests, logins=None, apikeys=None, test=0, cookies=None, token_type="Bearer"):
        super(APIMutateEnv, self).__init__()
        self.action_space           = spaces.Discrete(action_space)
        self.observation_space      = spaces.Box(0, 3000, (772, 1), dtype=int)
        
        self.default_token  = '***'
        self.previous_token = '***'
        self.requests       = request_seq
        self.token          = '***'
        self.current_request = self.requests[0]
        self.starting_request = self.current_request
        self.request_type = deepcopy(self.current_request.type)
        self.requests_idx = 0
        self.parameter_idx = 0
        self.api_session = requests.session()
        self.accessed_endpoints = []
        self.error_endpoints = []
        
        self.dynamic_param_table = DynamicStorage()
        self.login = {}
        self.logins = {}
        self.token_type = token_type
        self.interface = Interface(request_seq, logins, apikeys, cookies)
        self.parameter = self.current_request.parameters[self.parameter_idx] if len(self.current_request.parameters) > 0 else {'type': 'none', 'name': '', 'schema': {}}
        tokeniser_file = path.abspath(path.join(__file__, "../../../pre_processing/api_transformer/tokenizer.json"))
        self.tokeniser = PreTrainedTokenizerFast(tokenizer_file=tokeniser_file, padding_side='right',
                                            truncation_side='right', pad_token='0')
        self.feature_extractor = pipeline(
                "feature-extraction",
                model='m-foley/apirl-state-encoder',
                tokenizer='m-foley/apirl-state-encoder', 
                cache_dir=path.abspath(path.join(__file__, "../../pre_processing/api_transformer/"))
            )
        self.test = test
        self.test_body = {}
        self.req_200s = 0
        self.req_500s = 0
        self.bad_request = 0
        self.starting_observation =  self._generate_state(self.parameter, 000)
        self.state = self.starting_observation
        self.headers = {'Accept': 'application/json', "Content-Type": "application/json"}
        if cookies is not None or apikeys is not None:
            self.headers = {**self.headers, **self.interface.current_login}
        self.find_unauthed_request()
        
        self.generate_token()
        
        self.logins = {**self.interface.logins, **{'': ''}} 
        self.login = {list(self.logins)[0]: self.logins[list(self.logins)[0]]} if logins == None else {'': ''}
        self.param_locations = {}
        self.iterate_requests(all_requests)
        self.iterate_requests(all_requests)
        self.request_times = {}
        self.timer = time.time()
        self.previous_action = -1
        print('setup done')

    def extract_histories(self, endpoint):
        histories = ''
        for history in endpoint.history:
            status = history.status_code
            histories += str(status) + ' '
        histories = histories[:-1]
        return histories

    def _generate_state(self, value, status, endpoint=None):
        if 'schema' in value.keys() and not 'type' not in value['schema'].keys():
            value['type'] = value['schema']['type']
        if 'type' in value.keys():
            type = value['type']
        else:
            type = 'Unkown'
        var_type = None
        if re.search('str|text', type, re.I) is not None:
            var_type = 0
        elif re.search("int|num", type, re.I) is not None:
            var_type = 1
        elif  re.search('bool', type, re.I) is not None:
            var_type = 2
        elif type == 'object':
            var_type = 3
        elif type == 'array':
            var_type = 4
        elif type == 'Unkown':
            var_type = 5
        if var_type == None:
            var_type = 5
        if endpoint is None:
            response = [0 for i in range(768)]
        else:
            status_code = str(endpoint.status_code)
            headers = endpoint.headers
            if 'Date' in headers:
                headers.pop('Date')
            if 'Content-Type' in headers:
                headers.pop('Content-Type')
            if 'Content-Length' in headers:
                headers.pop('Content-Length')
            if 'Etag' in headers:
                headers.pop('Etag')
            if 'etag' in headers:
                headers.pop('etag')
            if 'Set-Cookie' in headers:
                headers['Set-Cookie'] = ''
            if 'Last-Modified' in headers:
                headers.pop('Last-Modified')
            if 'report-to' in headers:
                report_to = json.loads(headers['report-to'])
                if 'endpoints' in report_to:
                    for ep in range(len(report_to['endpoints'])):
                        report_to['endpoints'][ep]['url'] = report_to['endpoints'][ep]['url'].split('/')[2]
                headers['report-to'] = json.dumps(report_to)
            if 'report_to' in headers:
                headers.pop('Last-Modified')
            for header in endpoint.headers:
                headers[header] = re.sub('[^a-zA-Z -_]|\d', '', headers[header])
            reason = endpoint.reason
            encoding = endpoint.encoding
            apparent_encoding = endpoint.apparent_encoding
            history = self.extract_histories(endpoint)
            http = endpoint.url.split('/')[0][:-1]
            redirect = endpoint.is_redirect
            perma_redirect = endpoint.is_permanent_redirect
            response_as_string =str(status_code) + str(headers) + str(reason)+ str(encoding) +\
                                str(apparent_encoding)+ str(history) + str(http) + str(redirect) + str(perma_redirect)
            
            response = self.feature_extractor(response_as_string)[0][0]
            
                
        response = np.array(response)
        
        
        
        
        
        request_type = None
        if self.current_request.type == 'post':
            request_type = 0
        elif self.current_request.type == 'get':
            request_type = 1
        elif self.current_request.type == 'put':
            request_type = 2
        elif self.current_request.type == 'delete':
            request_type = 3
        elif self.current_request.type == 'patch':
            request_type = 4
        elif self.current_request.type == 'head':
            request_type = 5
        if any(vals == None for vals in [var_type, request_type]):
            print('unidentified state value:')
            print('req type:'+ str(request_type))
            
            print('var type:' + str(var_type))
        param_location = self.parameter_idx/len(self.current_request.parameters) if len(self.current_request.parameters) > 0 else 0
        
        return np.append(np.array([var_type, request_type, status, param_location]), response)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        
        
        
        self.state = self._generate_state(self.parameter, 000)
        
        self.current_request = deepcopy(self.requests[self.requests_idx])
        self.current_request.type = deepcopy(self.request_type)
        self.parameter_idx = 0
        
        self.parameter = self.current_request.parameters[self.parameter_idx] if len(self.current_request.parameters) > 0 else {'type': 'none', 'name': '', 'schema': {}, 'in': 'none'}
        if f'{self.current_request.path}_{self.current_request.type}' not in self.request_times:
            self.request_times[f'{self.current_request.path}_{self.current_request.type}'] = time.time() - self.timer
        else:
            self.request_times[f'{self.current_request.path}_{self.current_request.type}'] += time.time() - self.timer
        self.timer = time.time()
        return self.state

    def step(self, action):
        
        payload = self._create_payload(self.current_request.parameters)
        action_type, var = self._discrete_action_to_continuous(action)
        inj_var = None
        param_type = None
        parameter_name = None
        for param_location in payload:
            if 'name' in self.parameter.keys() and self.parameter['name'] in param_location.keys():
                parameter_value = deepcopy(param_location[self.parameter['name']])
                param_type =  deepcopy(self.parameter['schema']['type'] if 'schema' in self.parameter.keys() and not 'type' not in self.parameter['schema'].keys() else self.parameter['type'])
                parameter_name = None
                break
            if type(self.parameter) == dict and any(self.parameter['name']  == key for key in param_location.keys()):
                param_type = 'object'
                for key in param_location.keys():
                    if self.parameter['name'] == key:
                        parameter_name = deepcopy(random.choice(list(param_location[key].keys())))
                        parameter_value = deepcopy(param_location[key][parameter_name])
                
                break
        if param_type == None:
            print('param type is none')
        if len(self.current_request.parameters) == 0:
            print('removed all params')
        
        if action_type == 'increase' and param_type is not None:
            if re.search('int', param_type, re.I) is not None:
                if type(parameter_value) == int:
                    inj_var = parameter_value + var
                elif type(parameter_value) == str:
                    inj_var = parameter_value + str(var)
                else:
                    inj_var = parameter_value
            elif param_type == 'object':
                for sub_key, sub_val in parameter_value.items():
                    if any(char.isnumeric() for char in str(sub_val)):
                        for char in str(sub_val):
                            if char.isnumeric():
                                parameter_value[sub_key] = str(sub_val).replace(char, str(int(char) + var))
                                break
            elif any(char.isnumeric() for char in str(parameter_value)):
                for char in str(parameter_value):
                    if char.isnumeric():
                        inj_var = str(parameter_value).replace(char, str(int(char) + var))
                        break
            else:
                inj_var = parameter_value
            
        elif action_type == 'extension' and param_type is not None:
            if re.search('str', param_type, re.I):
                inj_var = '.'.join([str(parameter_value), var])
            elif re.search('object', param_type, re.I):
                inj_var = '.'.join([str(parameter_value[random.choice(list(parameter_value.keys()))]), var])
            else:
                inj_var = parameter_value
            
        elif action_type == 'wild' and param_type != None:
            if re.search('str', param_type, re.I) or re.search('int', param_type, re.I):
                
                inj_var = str(parameter_value)[:int(len(str(parameter_value))/2)] + var
                
                
            elif re.search('object', param_type, re.I):
                inj_var = str(parameter_value[random.choice(list(parameter_value.keys()))])[:int(len(str(parameter_value[random.choice(list(parameter_value.keys()))])) / 2)] + var
            else:
                inj_var = parameter_value
            
        elif action_type == 'append':
            try:
                if param_type == 'object':
                    header = random.choice(
                        [key for key in (parameter_value.keys()) if key in self.dynamic_param_table.keys])
                    possible_rows = self.dynamic_param_table.store[
                        self.dynamic_param_table.store[header].notnull()].sample(1)
                    chosen_val = possible_rows.loc[0][possible_rows.loc[0].notnull()].to_dict()
                    key = random.choice([key for key in chosen_val if key not in parameter_value.keys()])
                    val = chosen_val[key]
                    payload[2][self.parameter['name']][key] = val
                else:
                    inj_var = json.dumps(parameter_value)
                    param_header = self.dynamic_param_table.store[self.dynamic_param_table.store == parameter_value].stack().index[0][1]
                    
                    possible_params = self.dynamic_param_table.get_values_from_value({param_header: parameter_value})
                    key = random.choice([key for key in possible_params.keys() if key not in payload[2].keys()])
                    if param_type == 'array':
                        
                        inj_var.append(possible_params[key])
                    else:
                        payload[2][key] = possible_params[key]
                        
            except:
                print('append error')
                
            
            
        elif action_type == 'auth_token':
            self.interface.logged_in = False
            if var == 'refresh':
                if not self.generate_token(self.login):
                    if self.interface.current_login and re.search('Authorization|Cookie', str(list(self.interface.current_login.keys())[0])):
                        self.login = self.interface.current_login
                        self.headers = {**self.headers, **self.interface.current_login}
            elif var == 'switch':
                if len(self.logins) == 1:
                    if len(self.interface.apikey) > 0 and type(self.interface.apikey) != str:
                        new_login = [login for login in self.interface.apikey if login not in self.interface.current_login.values()][0]
                        self.interface.current_login = {'Authorization': new_login}
                    elif len(self.interface.cookie) > 0 and type(self.interface.cookie) != str:
                        new_login = [login for login in self.interface.cookie if login != self.interface.current_login][0]
                        self.interface.current_login = {'Cookies': new_login}
                    else:
                        self.generate_token()
                else:
                    current_login = list(self.login)[0]
                    new_login = [login for login in self.logins if login != current_login][0]
                    self.login = {new_login: self.logins[new_login]}
                    self.generate_token({new_login: self.logins[new_login]})
        elif action_type == 'param':
            same_request_paths = [request for request in self.requests if request.path == self.current_request.path and
                                  request.type != self.request_type and \
                                  not all(req_param in self.current_request.parameters for req_param in request.parameters)]
            if len(same_request_paths) > 0:
                alt_request = random.choice(same_request_paths)
                if len(alt_request.parameters) > 0:
                    add_param = random.choice([x for x in alt_request.parameters if x not in self.current_request.parameters])
                    add_payload = self._create_payload([add_param])
                    self.current_request.parameters.append(add_param)
                    self.parameter = self.current_request.parameters[-1]
                    self.parameter_idx = self.current_request.parameters.index(self.parameter)
                    payload = [{**add_payload[0], **payload[0]},
                               {**add_payload[1], **payload[1]},
                               {**add_payload[2], **payload[2]},
                               {**add_payload[3], **payload[3]}]
            elif self.parameter['name'] in self.dynamic_param_table.keys:
                keys = self.dynamic_param_table.get_related_keys(self.parameter['name'])
                add_param = []
                for key in keys:
                    if all(key not in param['name'] for param in self.requests[self.requests_idx].parameters):
                        if key != 'admin':
                            print('DEBUG')
                        type_list = [type(value) for value in self.dynamic_param_table.store[key].dropna().values]
                        most_common_type = str(max(type_list, key=type_list.count))
                        add_param.append({'name':key, 'in':'body', 'type':most_common_type})
                        
                        if {'name':key, 'in':'body', 'type':most_common_type} not in self.current_request.parameters:
                            self.current_request.parameters.append({'name':key, 'in':'body', 'type':most_common_type})
                            self.parameter = self.current_request.parameters[-1]
                            self.parameter_idx = self.current_request.parameters.index(self.parameter)
                add_payload = self._create_payload(add_param)
                payload = [{**add_payload[0], **payload[0]},
                           {**add_payload[1], **payload[1]},
                           {**add_payload[2], **payload[2]},
                           {**add_payload[3], **payload[3]}]
                
        elif action_type == 'request_method':
            self.current_request.type = var
        elif action_type == 'switch' and param_type is not None:
            
            self.parameter_idx = (self.parameter_idx + 1) % len(self.current_request.parameters) if len(self.current_request.parameters) > 0 else  0
            self.parameter     = self.current_request.parameters[self.parameter_idx]
            payload = self._create_payload(self.current_request.parameters)
        elif action_type == 'select' and param_type is not None:
            if self.parameter['name'] in self.dynamic_param_table.keys:
                possible_vals = self.dynamic_param_table.store[self.parameter['name']].values
                inj_var = random.choice(possible_vals)
                
            elif any(re.search(dynamic_param, self.parameter['name'], re.I)  for dynamic_param in self.dynamic_param_table.keys):
                for dynamic_param in self.dynamic_param_table.keys:
                    if re.search(dynamic_param, self.parameter['name'], re.I):
                        possible_vals =list(set(self.dynamic_param_table.store[dynamic_param].values)-set([None]))
                        if len(possible_vals) > 0:
                            inj_var = random.choice(possible_vals)
            elif any(re.search(self.parameter['name'], dynamic_param, re.I)  for dynamic_param in self.dynamic_param_table.keys):
                for dynamic_param in self.dynamic_param_table.keys:
                    if re.search(self.parameter['name'], dynamic_param, re.I):
                        possible_vals = list(set(self.dynamic_param_table.store[dynamic_param].values) - set([None]))
                        if len(possible_vals) > 0:
                            inj_var = random.choice(possible_vals)
        elif action_type == 'admin':
            payload = [payload[0],
                       payload[1],
                       {**{'admin': var}, **payload[2]},
                       payload[3]]
        elif action_type == 'whitelist':
            inj_var = var
        elif action_type == 'type' and param_type != None:
            
            if re.search('str', param_type, re.I):
                param_type = random.choice(['int', 'array', 'bool'])
            elif re.search('int', param_type, re.I):
                param_type = random.choice(['str', 'array'])
            elif re.search('bool', param_type, re.I):
                param_type = random.choice(['str'])
            elif re.search('array', param_type, re.I):
                param_type = random.choice(['int', 'str', 'bool'])
            if param_type == 'str':
                inj_var = str(parameter_value)
            elif param_type == 'bool':
                inj_var = random.choice([True, False])
            elif param_type == 'int':
                inj_var = random.choice([0, 1, -1])
            elif param_type == 'array':
                inj_var = [parameter_value]
        elif action_type == 'duplicate':
            duplicate_payload = self._create_payload([self.parameter])
            for sub_payload in duplicate_payload[1:]:
                if len(sub_payload) > 0:
                    
                    sub_payload[self.parameter['name']] = [sub_payload[self.parameter['name']],
                                                           sub_payload[self.parameter['name']]]
            payload = [{**payload[0], **duplicate_payload[0]},
                       {**payload[1], **duplicate_payload[1]},
                       {**payload[2], **duplicate_payload[2]},
                       {**payload[3], **duplicate_payload[3]}]
        elif action_type == 'remove':
            
            if len(self.current_request.parameters) != 0:
                self.current_request.parameters.pop(self.parameter_idx)
            if len(self.current_request.parameters) != 0:
                if self.parameter_idx > len(self.current_request.parameters) - 1:
                    self.parameter_idx = len(self.current_request.parameters) - 1
                self.parameter = self.current_request.parameters[self.parameter_idx]
            payload = self._create_payload(self.current_request.parameters)
            """inj_var = json.dumps(parameter_value)
            param_header = \
            self.dynamic_param_table.store[self.dynamic_param_table.store == parameter_value].stack().index[0][1]
            
            possible_params = self.dynamic_param_table.get_values_from_value({param_header: parameter_value})"""
            
        if inj_var is not None:
            for param_idx in range(len(payload)):
                if 'name' in self.parameter.keys() and self.parameter['name'] in payload[param_idx]:
                    if param_type == 'object' and random.random() < 0.75:
                        payload[2][self.parameter['name']][random.choice(list(payload[2][self.parameter['name']].keys()))] = inj_var
                    else:
                        payload[param_idx][self.parameter['name']] = inj_var
                elif type(self.parameter) == dict and any(
                        list(self.parameter)[0] == key for key in payload[param_idx].keys()) and parameter_name != None:
                    for key in payload[param_idx].keys():
                        if list(self.parameter)[0] == key:
                            payload[param_idx][key][parameter_name] = inj_var
        
        if self.interface.authed_request and self.interface.check_logged_in(self.interface.authed_request.parameters) == False:
            self.interface.logged_in = False
            self.generate_token(self.login)
        reward, done, status_code, response = self._compute_reward(payload, action_type, inj_var)
        state = self._generate_state(self.parameter, status_code, response)
        mutated_requests = None 
        info = {'action':(action_type, var), 'status':status_code,
                'method': self.current_request.type,
                'request': payload,
                'mutated_requests': mutated_requests}
        return np.array(state), reward, done, info
    

    def  _compute_reward(self, req_param, action_type, inj_var):
        
        
        header_params, path_params, body_params, query_params = req_param 
        header_params = {k:str(v) for k,v in header_params.items()}
        param_table = deepcopy(self.dynamic_param_table.store)
        table_keys = deepcopy(self.dynamic_param_table.keys)
        api_response = self._send_api_call(header_params, path_params, body_params, query_params)
        if api_response.status_code != 403 and api_response.status_code != 401 and re.search('not authori.ed|unauthori.ed', api_response.text, re.I) == None:
            self.accessed_endpoints.append(
                [deepcopy(self.current_request), copy(self.headers), api_response.request.url,
                 api_response.request.body, copy(self.login), api_response, self.requests[self.requests_idx]])
        new_table_keys = list(set(self.dynamic_param_table.keys) - set(table_keys))
        
        
        for new_key in new_table_keys:
            if new_key in api_response.text and new_key not in self.param_locations:
                        self.param_locations[new_key] = [self.current_request]
  
        if str(api_response.status_code)[0] == '5':
            self.error_endpoints.append([deepcopy(self.current_request), copy(self.headers), api_response.request.url, api_response.request.body, copy(self.login), api_response])
        if 200 <= api_response.status_code <=  299:
            self.req_200s += 1
            
            return (self.req_200s+self.req_500s)/(self.req_200s+self.bad_request+self.req_500s), 0, api_response.status_code, api_response
        elif 300 <= api_response.status_code <=  499:
            self.bad_request += 1
            return (self.req_200s+self.req_500s)/(self.req_200s+self.bad_request+self.req_500s), 0, api_response.status_code, api_response
        else:
            self.req_500s += 1
            return (self.req_200s+self.req_500s)/(self.req_200s+self.bad_request+self.req_500s), 1, api_response.status_code, api_response

    def iterate_requests(self, requests):
        all_parameters = [param  for req in requests for param in req.parameters]
        all_parameters = [json.loads(j) for j in list(set([json.dumps(i) for i in all_parameters]))]
        self.all_parameters = {}
        for param in all_parameters:
            if 'name' in param.keys():
                self.all_parameters[param['name']] = param
            else:
                self.all_parameters[list(param)[0]] = param
        
        for request in tqdm(requests, desc="populate param..."):
            if self.interface.authed_request and self.interface.check_logged_in(
                    self.interface.authed_request.parameters) == False:
                self.interface.logged_in = False
                self.generate_token(self.login)
            table_keys = copy(self.dynamic_param_table.keys)
            self.current_request = request
            header_params, path_params, body_params, query_params = self._create_payload(request.parameters)
            api_response = self._send_api_call(header_params, path_params, body_params, query_params)
            new_table_keys = list(set(self.dynamic_param_table.keys) - set(table_keys))
            for new_key in new_table_keys:
                if new_key in api_response.text:
                    if new_key not in self.param_locations:
                        self.param_locations[new_key] = [request]
            for param in all_parameters:
                if 'name' in param.keys() and param['name'] in api_response.text:
                    if param['name'] in self.param_locations:
                        if self.current_request.path not in self.param_locations[param['name']]:
                            self.param_locations[param['name']].append(request)
                    else:
                        self.param_locations[param['name']] = [request]
                elif list(param)[0] in api_response.text:
                    if list(param)[0] in self.param_locations:
                        if self.current_request.path not in self.param_locations[list(param)[0]]:
                            self.param_locations[list(param)[0]].append(request)
                    else:
                        self.param_locations[list(param)[0]] = [request]

    def find_unauthed_request(self):
        unprefered_requests = []
        for request in tqdm(self.requests, desc='finding unauthed request...'):
            self.current_request = request
            
            header_params, path_params, body_params, query_params = self._create_payload(request.parameters)
            api_response = self._send_api_call(header_params, path_params, body_params, query_params, parse_response=False)
            if request.type != 'delete' and api_response.status_code == 403 or api_response.status_code == 401 or re.search('not authori.ed|unauthori.ed', api_response.text, re.I) != None:
                if '{' in self.current_request.path:
                    unprefered_requests.append(request)
                else:
                    self.interface.authed_request = request
        if self.interface.authed_request == None and len(unprefered_requests) > 0:
            self.interface.authed_request = random.choice(unprefered_requests)

    def _create_payload(self, req_param):
        header_params   = {}
        path_params     = {}
        body_params     = {}
        query_params    = {}
        dynamic_param_id= None
        for param in req_param:
            if 'type' not in param.keys() and not ('schema' in param.keys() and not 'type' not in param['schema'].keys()) or ('type' in param.keys() and param['type'] == 'object'):
                if 'type' not in param.keys() and not ('schema' in param.keys() and not 'type' not in param['schema'].keys()):
                    print('Error param has no type:' + str(param))
                if 'sub_values' in param.keys():
                    _, _, sub_body_params, _ = self._create_payload(param['sub_values'])
                    
                    body_params[param['name']] = sub_body_params
                    
                    
                    continue
                else:
                    for key, sub_param in param.items():
                        if type(sub_param) == list:
                            _, _, sub_body_params, _ = self._create_payload(sub_param)
                            body_params[key] = sub_body_params
                            
                            
                    continue
            elif 'schema' in param.keys() and not 'type' not in param['schema'].keys():
                param['type'] = param['schema']['type']
            
            if 'inject' in param.keys():
                input_param = param['inject']
            elif re.search('int|number', param['type']):
                if random.choice([0,1]) == 0:
                    input_param = 0
                else:
                    input_param = 1
            elif  re.search('str', param['type']):
                if random.choice([0,1])  == 0 and re.search('register',self.current_request.path, re.I) is None:
                    input_param = ''
                else:
                    input_param = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            elif re.search('bool', param['type']):
                if random.choice([0,1])  == 0:
                    input_param = False
                else:
                    input_param = True
            elif re.search('array|list', param['type']):
                if 'sub_type' in param.keys():
                    if re.search('int', param['sub_type']):
                        if random.choice([0, 1]) == 0:
                            input_param = [0]
                        else:
                            input_param = [1]
                    elif re.search('str', param['sub_type']):
                        if random.choice([0, 1]) == 0 and re.search('register', self.current_request.path, re.I) is None:
                            input_param = ['']
                        else:
                            input_param = [''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))]
                    elif re.search('bool', param['sub_type']):
                        if random.choice([0, 1]) == 0:
                            input_param = [False]
                        else:
                            input_param = [True]
                    elif re.search('object', param['sub_type']):
                        if random.choice([0, 1]) == 0:
                            input_param = [{}]
                        else:
                            input_param = []
                    else:
                        print('hmm')
                else:
                    if random.choice([0, 1])  == 0:
                        input_param = ['']
                    else:
                        input_param = []
            
            if 'example' in param.keys():
                input_param = param['example']
            elif 'default' in param.keys():
                input_param = param['default']
            elif 'email' in param['name']:
                input_param = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8)) + '@' + \
                              ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8)) + '.com'
            elif 'password' in param['name']:
                input_param = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            """
            if re.search('name|username|uname|user', param['name'], re.I) and login is not None:
                input_param = login[0]
            elif re.search('auth|token', param['name'], re.I) and login is not None:
                input_param = login[1]
            elif re.search('password|pword', param['name'], re.I) and login is not None:
                input_param = login[2]"""
            
            
            
            
            
            random_chance =  random.random() < 0.9
            if (re.search('register', self.current_request.path) is None and random_chance) and (re.search('register', self.current_request.summary, re.I) is None and random.random() <  random_chance):
                if any(re.search(dynamic_param, param['name'], re.I)  for dynamic_param in self.dynamic_param_table.keys):
                    try:
                        if dynamic_param_id is None:
                            if 'user' in self.dynamic_param_table.keys:
                                dynam_input_usr = list(self.logins)[0]
                                
                                
                                
                            elif 'username' in self.dynamic_param_table.keys:
                                dynam_input_usr = list(self.logins)[0]
                                
                                
                                
                            elif 'name' in self.dynamic_param_table.keys:
                                dynam_input_usr = list(self.logins)[0]
                                
                                
                                
                            dynamic_param_id = self.dynamic_param_table.get_id_from_val({'username':dynam_input_usr})
                            possible_id = self._check_params(req_param)
                            dynamic_param_id = possible_id if possible_id is not None else dynamic_param_id
                            dynam_input_param = self.dynamic_param_table.store.loc[dynamic_param_id, param['name']]
                        else:
                            dynam_input_param = self.dynamic_param_table.store.loc[dynamic_param_id, param['name']]
                        if dynam_input_param is None:
                            dynamic_param = \
                            list(filter(lambda x: re.search(x, param['name'], re.I), self.dynamic_param_table.keys))[0]
                            if dynamic_param_id is None:
                                possible_params = list(
                                    set(self.dynamic_param_table.store[dynamic_param].values) - set([None]))
                                if len(possible_params) > 0:
                                    dynam_input_param = random.choice(possible_params)
                                    dynamic_param_id = self.dynamic_param_table.get_id_from_val(
                                        {dynamic_param: dynam_input_param})
                                    possible_id = self._check_params(req_param)
                                    dynamic_param_id = possible_id if possible_id is not None else dynamic_param_id
                            else:
                                dynam_input_param = self.dynamic_param_table.store.loc[dynamic_param_id, dynamic_param]
                    except:
                        dynamic_param = list(filter(lambda x: re.search(x, param['name'], re.I), self.dynamic_param_table.keys))[0]
                        if dynamic_param_id is None:
                            possible_params = list(set(self.dynamic_param_table.store[dynamic_param].values)-set([None]))
                            if len(possible_params) > 0:
                                dynam_input_param = random.choice(possible_params)
                                dynamic_param_id = self.dynamic_param_table.get_id_from_val({dynamic_param:dynam_input_param})
                                possible_id = self._check_params(req_param)
                                dynamic_param_id = possible_id if possible_id is not None else dynamic_param_id
                            else:
                                dynam_input_param = None
                        else:
                            dynam_input_param = self.dynamic_param_table.store.loc[dynamic_param_id, dynamic_param]
                    if dynam_input_param is not None:
                        input_param = dynam_input_param
                elif any(re.search(re.escape(param['name']), dynamic_param, re.I)  for dynamic_param in self.dynamic_param_table.keys):
                    for dynamic_param in self.dynamic_param_table.keys:
                        if re.search(param['name'], dynamic_param, re.I) is not None:
                            param_id = dynamic_param
                            break
                    dynamic_param = list(filter(lambda x: re.search(param['name'], x, re.I), self.dynamic_param_table.keys))[0]
                    if dynamic_param_id is None:
                        possible_params = list(set(self.dynamic_param_table.store[dynamic_param].values) - set([None]))
                        if len(possible_params) > 0:
                            dynam_input_param = random.choice(possible_params)
                            dynamic_param_id = self.dynamic_param_table.get_id_from_val({dynamic_param: dynam_input_param})
                            possible_id = self._check_params(req_param)
                            dynamic_param_id = possible_id if possible_id is not None else dynamic_param_id
                        else:
                            dynam_input_param = None
                    else:
                            dynam_input_param = self.dynamic_param_table.store.loc[dynamic_param_id, dynamic_param]
                    if dynam_input_param is not None:
                        input_param = dynam_input_param
            if param['in'] == 'path':
                path_params[param['name']] = input_param
            if param['in'] == 'query':
                query_params[param['name']] = input_param
            elif param['in'] == 'body':
                
                
                
                body_params[param['name']] = input_param
            elif param['in'] == 'header':
                if 'default' in param.keys():
                    header_params[param['name']] = str(param['default'])
                
                
                else:
                    header_params[param['name']] = str(input_param)
        
        
        
        
        return header_params, path_params, body_params, query_params

    def _check_params(self, params):
        values = self.dynamic_param_table.store
        dynamic_params = []
        for param in params:
            if 'name' not in param:
                possible_params = []
                for key, sub_param in param.items():
                    if type(sub_param) == list:
                        possible_params.append(self._check_params(sub_param))
                return random.choice(possible_params)
            if param['name'] in values.columns:
                dynamic_params.append(param['name'])
            else:
                try:
                    dynamic_params.append(list(filter(lambda x: re.search(x, param['name'], re.I), self.dynamic_param_table.keys))[0])
                except:
                    try:
                        dynamic_params.append(list(filter(lambda x: re.search(param['name'], x, re.I), self.dynamic_param_table.keys))[0])
                    except:
                        print('param not in table...')
        try:
            for key in dynamic_params:
                values = values[~(values[key].apply(lambda x: str(x) == 'None'))]
        except:
            pass
        idx = None
        if values.shape[0] != 0:
            idx = random.choice(values.index.values)
        return idx
    

    def _send_api_call(self, header_params, get_params, body_params, query_params, parse_response=True):
        
        
        api_path = self.current_request.path
        
        param_keys = list(get_params.keys())
        path_call_params = {}
        for param in param_keys:
            if ''.join(['{', param, '}']) in api_path:
                api_path = api_path.replace(''.join(['{', param, '}']), str(get_params[param]))
                path_call_params.update({param:get_params[param]})
                get_params.pop(param)
        if len(query_params) > 0:
            query = ''
            for key, value in query_params.items():
                query += str(key) + '=' + str(value) + '&'
            api_path = api_path + '?' + query[:-1]
        
        print(json.dumps(body_params))
        
        print({**self.headers, ** header_params})
        if self.current_request.type == 'delete':
            response = self.api_session.delete(api_path, data=json.dumps(body_params),
                                       params=get_params,
                                       headers={**self.headers, ** header_params})
        if self.current_request.type == 'get':
            response = self.api_session.get(api_path, data=json.dumps(body_params),
                                    params=get_params,
                                    headers={**self.headers, ** header_params})
        if self.current_request.type == 'put':
            response = self.api_session.put(api_path, data=json.dumps(body_params),
                                    params=get_params,
                                    headers={**self.headers, ** header_params})
        if self.current_request.type == 'post':
            response = self.api_session.post(api_path, data=json.dumps(body_params),
                                     params=get_params,
                                     headers={**self.headers, ** header_params})
        if self.current_request.type == 'patch':
            response = self.api_session.patch(api_path, data=json.dumps(body_params),
                                     params=get_params,
                                     headers={**self.headers, ** header_params})
        if self.current_request.type == 'head':
            response = self.api_session.head(api_path, data=json.dumps(body_params),
                                     params=get_params,
                                     headers={**self.headers, ** header_params})
        
        print(api_path, self.current_request.type, response)
        if parse_response:
            self._parse_response(response, {**body_params, **header_params, **get_params, **path_call_params})
        if any(re.search('X-RateLimit-Remaining', header, re.I) is not None for header in response.headers):
            for header in response.headers:
                if re.search('X-RateLimit-Remaining', header, re.I) is not None:
                    requests_remaining = response.headers[header]
                    break
            if int(requests_remaining) == 0:
                for header in response.headers:
                    if re.search('X-RateLimit-Reset', header, re.I) is not None:
                        time_remaining = response.headers[header]
                        break
                time.sleep(int(time_remaining))
        elif response.status_code == 429:
            print('too many requests')
        if re.search('put|post', self.current_request.type) is not None and parse_response:
            param_update_request = self.api_session.get(api_path, data=json.dumps(body_params),
                                    params=get_params,
                                    headers={**self.headers, **header_params})
            self._parse_response(param_update_request, {**body_params, **header_params, **get_params, **path_call_params})
        return response
    

    def _discrete_action_to_continuous(self, discrete_action):
        
        if discrete_action == 0:
            action = ('increase', 1)
        
        elif discrete_action == 1:
            action = ('increase', -1)
        
        elif discrete_action in range(2, 5):
            extensions = ['txt', 'pdf', 'doc']
            action = ('extension', extensions[discrete_action - 2])
        
        elif discrete_action in range(5, 8):
            wild_cards = ['*', '.*', '%']
            action = ('wild', wild_cards[discrete_action - 5])
        
        elif discrete_action == 8:
            action = ('append', '')
        
        elif discrete_action == 9:
            action = ('auth_token', 'refresh')
        
        elif discrete_action == 10:
            
            action = ('param', '')
        
        elif discrete_action == 11:
            if self.current_request.type == 'put':
                action = ('request_method', 'post')
            elif self.current_request.type == 'post':
                action = ('request_method', 'put')
            else:
                action = ('request_method', self.current_request.type)
        elif discrete_action == 12: 
            action = ('switch', 1)
        elif discrete_action == 13:
            
            action = ('select', '')
            
        elif discrete_action == 14:
            action = ('admin', True)
        elif discrete_action == 15:
            action = ('auth_token', 'switch')
        elif discrete_action in range(16, 20):
            whitelist_params = ['admin', -1, 99999999, '']
            action = ('whitelist', whitelist_params[discrete_action - 16])
        
        
        
        elif discrete_action == 20:
            
            action = ('type', '')
        elif discrete_action == 21:
            action = ('duplicate', '')
        elif discrete_action == 22:
            action = ('remove', '')
        
        else:
            print('hmm')
        return action
        
    

    def url_decode(self, state):
        return urllib.parse.unquote(state)

    def json_decode(self, state):
        return json.loads(state)

    def html_decode(self, state):
        return html.unescape(state)

    def _parse_tree(self, tree):
        leaf_set = set()
        if type(tree) == list:
            for leaf in tree:
                if type(leaf) == list:
                    leaf_set.update(self._parse_tree(tree[leaf]))
                elif type(leaf) == dict:
                    
                    
                    new_leaf_set, value_added = self._dict_insert(leaf)
                    if new_leaf_set != None:
                        leaf_set.update(new_leaf_set)
        if type(tree) == dict:
            value_added = False
            for leaf, leaf_value in tree.items():
                if type(leaf_value) == list:
                    value_added = True
                    leaf_set.update(self._parse_tree(tree[leaf]))
                elif type(leaf_value) == dict:
                    new_leaf_set, value_added = self._dict_insert(leaf_value)
                    if new_leaf_set != None:
                        leaf_set.update(new_leaf_set)
                else:
                    try:
                        leaf not in self.dynamic_param_table.keys
                    except:
                        continue 
                    if leaf not in self.dynamic_param_table.keys: 
                            
                        self.dynamic_param_table.add_column(leaf)
            if value_added == False and any(self.dynamic_param_table.in_table(leaf_value) == False for leaf_value in tree.values()):
                relation_value = False
                relations = {}
                for leaf, leaf_value in tree.items():
                    if self.dynamic_param_table.in_table(leaf_value) == True:
                        if leaf not in self.dynamic_param_table.keys:
                            for column in self.dynamic_param_table.store.keys():
                                if self.dynamic_param_table.store[self.dynamic_param_table.store[column] == leaf_value].shape[
                                    0] > 0:
                                    leaf = column
                                    break
                        
                        
                        try:
                            leaf_value is not None and self.dynamic_param_table.store[self.dynamic_param_table.store[leaf] == leaf_value].shape[0] > 0
                        except:
                            try:
                                leaf_value = leaf_value[0]
                            except:
                                __import__("IPython").embed()
                        if leaf_value is not None and self.dynamic_param_table.store[self.dynamic_param_table.store[leaf] == leaf_value].shape[
                                    0] > 0:
                            if leaf in relations.keys() and leaf in tree and leaf_value != tree[leaf]:
                                print('bruh')
                            elif leaf in tree and leaf_value != tree[leaf]:
                                print('jkb')
                            relation_value = {leaf: leaf_value}
                            relations[leaf] = leaf_value
                            if leaf in tree and leaf_value != tree[leaf]:
                                print('wtf')
                        
                if relation_value != False and self.dynamic_param_table.in_table(relation_value[list(relation_value)[0]]):
                    to_pop = []
                    for leaf_key, leave_value in tree.items():
                        if type(leave_value) == dict or type(leave_value) == list:
                            self._parse_tree(leave_value)
                            to_pop.append(leaf_key)
                    for popable in to_pop:
                        tree.pop(popable)
                    if len(relations) > 1:
                        indices = []
                        
                        for key, value in relations.items():
                            if value is not None:
                                indices.append(
                                    self.dynamic_param_table.store.loc[self.dynamic_param_table.store[key] == value].index[
                                        0])
                        mode = statistics.mode(indices)
                        idx = indices.index(mode)
                        relation_value = {list(relations)[idx]: relations[list(relations)[idx]]}
                    for leaf, leaf_value in tree.items():
                        if leaf_value not in self.dynamic_param_table.store.loc[
                            self.dynamic_param_table.store[list(relation_value)[0]] == relation_value[
                                list(relation_value)[0]]].values or \
                                leaf_value not in self.dynamic_param_table.store.loc[:,list(relation_value)[0]]:
                            if leaf in self.dynamic_param_table.keys:
                                dynamic_key = leaf
                            else:
                                dynamic_key = \
                                list(filter(lambda x: re.search(leaf, x, re.I), self.dynamic_param_table.keys))[0]
                            if type(leaf_value) in [list, dict]:
                                self._parse_tree(leaf_value)
                            else:
                                self.dynamic_param_table.insert_value({dynamic_key: leaf_value}, relation_value)
                else:
                    for key, value in tree.items():
                        if key not in self.dynamic_param_table.keys:
                            self.dynamic_param_table.add_column(key)
                    add_as_row = True
                    for leaf_key, leave_value in tree.items():
                        if type(leave_value) != string or type(leave_value) != int:
                            add_as_row = False
                            break
                    if add_as_row == True:
                        self.dynamic_param_table.add_row(tree)
                    else:
                        keys_to_pop = []
                        for key, leaf in tree.items():
                            if type(leaf) == dict or type(leaf) == list:
                                self._parse_tree(leaf)
                                keys_to_pop.append(key)
                        for key in keys_to_pop:
                            tree.pop(key)
                        self.dynamic_param_table.add_row(tree)
        return leaf_set

    def _dict_insert(self, leaf_value):
        leaf_set = set()
        
        value_added = True
        for key, value in leaf_value.items():
            if key not in self.dynamic_param_table.keys:  
                
                self.dynamic_param_table.add_column(key)
        keys_to_pop = []
        for key, leaf in leaf_value.items():
            if type(leaf) == dict or type(leaf) == list:
                self._parse_tree(leaf)
                keys_to_pop.append(key)
        for key in keys_to_pop:
            leaf_value.pop(key)
        if any(self.dynamic_param_table.in_table(value) == False for value in leaf_value.values()):
            relation_value = False
            relations = {}
            for key, value in leaf_value.items():
                if self.dynamic_param_table.in_table(value) == True:
                    
                    
                    
                    for column in self.dynamic_param_table.store.keys():
                        if self.dynamic_param_table.store[self.dynamic_param_table.store[column] == value].shape[0] > 0 \
                                and key not in self.dynamic_param_table.keys:
                            key = column
                            break
                            
                    if value is not None and key in self.dynamic_param_table.keys and self.dynamic_param_table.store[self.dynamic_param_table.store[key] == value].shape[0] > 0:
                        relation_value = {key: value}
                        relations[key] = value
                    
            if relation_value != False:
                if len(relations) > 1:
                    indices = []
                    for key, value in relations.items():
                        indices.append(self.dynamic_param_table.store.loc[
                                           self.dynamic_param_table.store[key] == value].index[0])
                    try:
                        mode = statistics.mode(indices)
                    except:
                        
                        mode = max([p[0] for p in statistics._counts(indices)])
                    idx = indices.index(mode)
                    relation_value = {list(relations)[idx]: relations[list(relations)[idx]]}
                for items, value in leaf_value.items():
                    if type(value) == dict or type(value) == list:
                        self._parse_tree(value)
                        continue
                    if value not in self.dynamic_param_table.store.loc[
                        self.dynamic_param_table.store[list(relation_value)[0]] == relation_value[
                            list(relation_value)[0]]].values or \
                            self.dynamic_param_table.store.loc[self.dynamic_param_table.store.loc
                                                               [self.dynamic_param_table.store[
                                                                    list(relation_value)[0]] == relation_value[
                                                                    list(relation_value)[0]]].index[0], items] != value:
                        dynamic_key = \
                            list(filter(lambda x: re.search(items, x, re.I), self.dynamic_param_table.keys))[0]
                        if self.dynamic_param_table.store[self.dynamic_param_table.store[list(relation_value)[0]] == relation_value[list(relation_value)[0]]].shape[0] == 0:
                            relations.pop(list(relation_value)[0])
                            relation_key = random.choice(list(relations.keys()))
                            relation_value = {relation_key:relations[relation_key]}
                        self.dynamic_param_table.insert_value({dynamic_key: value}, relation_value)
                        
                        
                        
                        
            else:
                keys_to_pop = []
                for key, leaf in leaf_value.items():
                    if type(leaf) == dict or type(leaf) == list:
                        self._parse_tree(leaf)
                        keys_to_pop.append(key)
                for key in keys_to_pop:
                    leaf_value.pop(key)
                self.dynamic_param_table.add_row(leaf_value)
                
                
                
                
            if leaf_set == None:
                print('as')
            return leaf_set, value_added
        else:
            return None, False

    def _parse_response(self, response, request_params, parse_for_auth_token=False):
        if response.text is not None and response.text != '':
            try:
                response_params = json.loads(response.text)
            except:
                if response.status_code == 500:
                    return
                else:
                    print('i shouldn\'t be here')
                response_params = BeautifulSoup(response.text, 'html.parser')
                try:
                    response_params = json.loads(response_params.text)
                except:
                    return
            
            response_set = self._parse_tree(response_params)
            if parse_for_auth_token:
                try:
                    dynamic_param = list(filter(lambda x: re.search('auth|token',x, re.I), self.dynamic_param_table.keys))[0]
                    response_params = {key.split('.')[-1]: value for key, value in pd.json_normalize(response_params, record_prefix="").to_dict(orient='records')[0].items()}
                    
                    self.headers['Authorization'] = ' '.join([self.token_type, response_params[dynamic_param]])
                    self.interface.headers = self.headers
                except:
                    try:
                        if type(response_params) == str and re.search('login|sigin|auth|token', response.url, re.I):
                            self.headers['Authorization'] = ' '.join([self.token_type, response_params])
                            self.interface.headers = self.headers
                    except:
                        pass
                    pass
        if (response.status_code == 200 and re.search('login', response.request.url, re.I) and not re.search('fail|error|not exist', response.text)) or response.status_code == 204:
            if len(list(filter(lambda x: re.search(x, 'user|username|uname', re.I), request_params.keys()))) > 0:
                username_field = list(filter(lambda x: re.search(x, 'user|username|uname', re.I), request_params.keys()))[0]
            elif len(list(filter(lambda x: re.search(x, 'email', re.I), request_params.keys())))> 0:
                username_field = list(filter(lambda x: re.search(x, 'email', re.I), request_params.keys()))[0]
            else:
                username_field = None
            if len(list(filter(lambda x: re.search(x, 'password|pass|pword', re.I), request_params.keys()))) > 0:
                password_field = list(filter(lambda x: re.search(x, 'password|pass|pword', re.I), request_params.keys()))[0]
            else:
                password_field = None
            if 'Authorization' in self.headers.keys():
                
                if password_field is not None:
                    self.login = {'': ''}
                    self.login = {request_params[username_field]: request_params[password_field]}
                    self.interface.current_login = self.login
                    self.logins = {**self.logins, **self.login}
                    self.interface.logins = self.logins
                else:
                    self.login = self.login
            
            
        elif response.status_code == 200 and re.search('logout|sign*out', response.request.url, re.I):
            self.login ={'': ''}


    def generate_token(self, user_login=None):
        if user_login != None:
            try:
                response, request_params = self.interface.user_login(user_login)
                if len(self.logins) > 0:
                    self._parse_response(response, request_params, parse_for_auth_token=True)
                else:
                    self.headers = {**self.headers, **response}
                return True
            except:
                self.interface.user_login(user_login)
                return False
        else:
            try:
                response, request_params = self.interface.user_login()
                if len(list(self.logins)) > 0 or type(response) == Response:
                    self._parse_response(response, request_params, parse_for_auth_token=True)
                else:
                    self.headers = {**self.headers, **response}
                return True
            except:
                print('could not login')
                self.interface.user_login()
                return False
