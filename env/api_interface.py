import random
import string
import re
import json
import requests
from copy import deepcopy
from datetime import datetime


class Interface:
    def __init__(self, requests_seq, login=None, apikey=None, cookies=None):
        #self.login = login
        self.apikey = apikey
        self.session = requests.session()
        self.current_login = None

        if login == None:
            self.logins = {}
        else:
            self.current_login = {list(login)[0]: login[list(login)[0]]}
            self.logins = login
        if apikey is None:
            self.apikey = {}
        else:
            self.apikey = apikey
            if type(apikey) == dict:
                self.current_login = {list(self.apikey)[0]: self.apikey[list(self.apikey)[0]]}
                self.apikey = self.apikey[list(self.apikey)[0]]
            else:
                self.current_login = {'Authorization': self.apikey[0]}
        if cookies is None:
            self.cookie = {}
        else:
            self.cookie = cookies
            self.current_login = {'Cookie': cookies[0]}
        self.logged_in = False
        self.requests_seq = deepcopy(requests_seq)
        self.current_request = self.requests_seq[0]
        self.previous_request = None
        self.headers ={'Accept': 'application/json', "Content-Type": "application/json"}
        self.authed_request = None


    def user_login(self, login=None):
        if self.authed_request == None:
            return True
        #login_idx = None if login == None else list(self.logins).index(list(login)[0])
        register_request = [req for req in self.requests_seq if re.search('register|create', req.path, re.I) or re.search('register', req.summary, re.I)]
        print('logging in...')
        if len(register_request) > 0:
            register_request = register_request[0]
        if type(self.current_login) !=  dict and (self.current_login is None or self.current_login[0] == 0) and login == None and type(register_request) != list:
            body_params = {}
            username = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
            password = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
            if type(register_request) == list:
                print('how')
            for param in register_request.parameters:
                input_param = self._get_basic_parameter(param)
                if param['type'] == 'object':
                    for sub_param in param['sub_values']:
                        if re.search('name|username|uname|user', sub_param['name'], re.I):
                            input_param[param['name']][sub_param['name']] = username
                        elif re.search('password|pword', sub_param['name'], re.I):
                            input_param[param['name']][sub_param['name']] = password
                        elif re.search('email', sub_param['name'], re.I):
                            input_param[param['name']][sub_param['name']] = ''.join([username, '@email.com'])
                    body_params = input_param
                else:
                    if re.search('name|username|uname|user', param['name'], re.I):
                        input_param = username
                    elif re.search('password|pword', param['name'], re.I):
                        input_param = password
                    elif re.search('email', param['name'], re.I):
                        input_param = ''.join([username, '@email.com'])
                    body_params[param['name']] = input_param
            reg_response = self.session.post(register_request.path, data=json.dumps(body_params),
                                             headers=self.headers)
            if str(reg_response.status_code)[0] == '2':
                login_request = [req for req in self.requests_seq if re.search('login|sigin', req.path, re.I)][0]
                login_body = {}
                for param in login_request.parameters:
                    input_param = self._get_basic_parameter(param)
                    if param['type'] == 'object':
                        for sub_param in param['sub_values']:
                            if re.search('name|username|uname|user', sub_param['name'], re.I):
                                input_param[param['name']][sub_param['name']] = username
                            elif re.search('password|pword', sub_param['name'], re.I):
                                input_param[param['name']][sub_param['name']] = password
                            elif re.search('email', sub_param['name'], re.I):
                                input_param[param['name']][sub_param['name']] = ''.join([username, '@email.com'])
                        login_body = input_param
                    else:
                        if re.search('name|username|uname|user', param['name'], re.I):
                            input_param = username
                        elif re.search('password|pword', param['name'], re.I):
                            input_param = password
                        elif re.search('email', param['name'], re.I):
                            input_param = ''.join([username, '@email.com'])
                        login_body[param['name']] = input_param

                if login_request.type == 'get':
                    login_response = self.session.get(login_request.path, data=json.dumps(login_body),
                                                   headers=self.headers)
                else:
                    login_response = self.session.post(login_request.path, data=json.dumps(login_body),
                                                   headers=self.headers)
                #self._parse_response(login_response, login_body)
                if login_response.status_code == 200 or self.current_login is None or self.current_login[0] == 0:
                    self.logins[username] = password
                    self.current_login = {username: password}
                    return login_response, login_body

            else:
                print('registration unsuccessful')

        if login and len([req for req in self.requests_seq if re.search('login|sigin|auth|token', req.path, re.I)]) > 0:
            body_params = {}
            user_login = (list(login)[0], login[list(login)[0]])
            login_requests = [req for req in self.requests_seq if re.search('login|sigin|auth|token', req.path, re.I)]
            for login_request in login_requests:
                username = user_login[0]
                password = user_login[1]
                if type(register_request) == list:
                    register_request = login_request
                for param in register_request.parameters:
                    input_param = self._get_basic_parameter(param)
                    if type(input_param) == list:
                        body_params[list(param)[0]] = input_param
                    elif re.search('name|username|uname|user|email', param['name'], re.I) and user_login is not None:
                        input_param = user_login[0]
                    elif re.search('password|pword', param['name'], re.I) and user_login is not None:
                        input_param = user_login[1]
                    if 'name' in param:
                        body_params[param['name']] = input_param

                if login_request.type == 'get':
                    response = self.session.get(login_request.path, data=json.dumps(body_params),
                                             headers=self.headers)
                else:
                    response = self.session.post(login_request.path, data=json.dumps(body_params),
                                             headers=self.headers)

                if 200 <= response.status_code <= 299 and re.search('fail(ed)?|reject(ed)?|incomplete', response.text) == None:
                    self.logged_in = True
                    # self._parse_response(login_response, login_body)
                    self.current_login = {username: password}
                    return response, body_params

        if len(self.logins) > 0:
            #login_idx = None if login == None or list(login)[0] not in self.logins.keys() else list(self.logins).index(list(login)[0])
            login_idx = 0 #if login_idx == None else login_idx
            #user_login = (list(self.logins)[login_idx], self.logins[list(self.logins)[login_idx]])
            #username = user_login[0]
            #password = user_login[1]
            login_request = [req for req in self.requests_seq if re.search('login|sigin|auth|token', req.path, re.I)]
            if len(login_request) == 1:
                login_requests = login_request
            elif len(login_request) == 0:
                return
            else:
                login_requests = login_request


            while self.logged_in == False:
                user_login = (list(self.logins)[login_idx], self.logins[list(self.logins)[login_idx]])
                username = user_login[0]
                password = user_login[1]
                for login_request in login_requests:
                    if user_login[0] != 0 and user_login[1] != 0:
                        body_params = {}
                        for param in login_request.parameters:
                            input_param = self._get_basic_parameter(param)
                            if param['type'] == 'object':
                                for sub_param in param['sub_values']:
                                    if re.search('name|username|uname|user|email', sub_param['name'], re.I) and user_login is not None:
                                        input_param[param['name']][sub_param['name']] = user_login[0]
                                    elif re.search('password|pword', sub_param['name'], re.I) and user_login is not None:
                                        input_param[param['name']][sub_param['name']] = user_login[1]
                                body_params = input_param
                            else:
                                if re.search('name|username|uname|user|email', param['name'], re.I) and user_login is not None:
                                    input_param = user_login[0]
                                elif re.search('password|pword', param['name'], re.I) and user_login is not None:
                                    input_param = user_login[1]
                                body_params[param['name']] = input_param
                        if login_request.type == 'post':
                            response = self.session.post(login_request.path, data=json.dumps(body_params),
                                                     headers=self.headers)
                        if login_request.type == 'get':
                            response = self.session.get(login_request.path, data=json.dumps(body_params),
                                                     headers=self.headers)
                        if 200 <= response.status_code <= 299 and re.search('fail(ed)?|reject(ed)?|incomplete', response.text) == None:
                            self.logged_in = True
                            #self._parse_response(login_response, login_body
                            self.current_login = {username: password}
                            return response, body_params
                        #elif 'token' in login_request.path:
                        #    print('error')
                        #    print(response)
                        #    print(response.text)

                login_idx += 1
                if login_idx > len(self.logins) - 1:
                    #if self.current_login is None and len(self.logins) > 0:
                    if 'Cookie' not in self.current_login.keys() and 'Authorization' and self.current_login.keys():
                        self.current_login = None
                        return self.user_login()
                    else:
                        return

                user_login = (list(self.logins)[login_idx], self.logins[list(self.logins)[login_idx]])
        if len(self.apikey) > 0 or len(self.cookie) > 0:
            user_login = self.current_login


            if len(user_login) > 0:
                body_params = {}
                if self.authed_request.type == 'post':
                    response = self.session.post(self.authed_request.path, data=json.dumps(body_params),
                                             headers={**self.headers, **user_login})
                if self.authed_request.type == 'get':
                    response = self.session.get(self.authed_request.path, data=json.dumps(body_params),
                                                 headers={**self.headers, **user_login})
                if self.authed_request.type == 'put':
                    response = self.session.put(self.authed_request.path, data=json.dumps(body_params),
                                                 headers={**self.headers, **user_login})
                if self.authed_request.type == 'patch':
                    response = self.session.patch(self.authed_request.path, data=json.dumps(body_params),
                                                 headers={**self.headers, **user_login})
                if self.authed_request.type == 'head':
                    response = self.session.head(self.authed_request.path, data=json.dumps(body_params),
                                                 headers={**self.headers, **user_login})

                #if 200 <= response.status_code <= 299 and re.search('fail(ed)?|reject(ed)?|incomplete', response.text) == None:
                #    self.logged_in = True
                #    #self._parse_response(login_response, login_body)
                #    self.current_login = (list(self.apikey)[login_idx], self.logins[list(self.apikey)[login_idx]])
                #    return response, body_params
                return user_login, {}

    def check_logged_in(self, params):
        header_params, path_params, body_params, query_params = self._create_payload(params)
        response = self._send_api_call(header_params, path_params, body_params, query_params)
        logged_in = False
        if (response.status_code != 403 and response.status_code != 401) and re.search('not authori.ed|unauthori.ed|invalid', response.text, re.I) == None:
            logged_in = True
        return logged_in


    #header_params, path_params, body_params
    def _send_api_call(self, header_params, get_params, body_params, query_params):
        # get api path
        #api_path = ''.join(['http://', self.authed_request.path])
        api_path = self.authed_request.path
        # replace dynamic parameters into the path of the api call
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

        if self.authed_request.type == 'get':
            response = self.session.get(api_path, data=json.dumps(body_params),
                                    params=get_params,
                                    headers={**self.headers, ** header_params})
        if self.authed_request.type == 'put':
            response = self.session.put(api_path, data=json.dumps(body_params),
                                    params=get_params,
                                    headers={**self.headers, ** header_params})
        if self.authed_request.type == 'post':
            response = self.session.post(api_path, data=json.dumps(body_params),
                                     params=get_params,
                                     headers={**self.headers, ** header_params})
        if self.authed_request.type == 'patch':
            response = self.session.patch(api_path, data=json.dumps(body_params),
                                     params=get_params,
                                     headers={**self.headers, ** header_params})
        return response

    def _get_basic_parameter(self, parameter):
        if type(parameter) == list:
            input_param = []
            for param in parameter:
                input_param.append({param['name']: self._get_basic_parameter(param)})
            return input_param
        if 'type' not in parameter.keys() and ('schema' in parameter.keys() and 'type' in parameter['schema'].keys()):
            parameter['type'] = parameter['schema']['type']
        if 'type' not in parameter or parameter['type'] == 'object':
            if 'sub_values' in parameter.keys():
                input_param = {}
                for param in parameter['sub_values']:
                    input_param = {**{param['name']:self._get_basic_parameter(param)}, **input_param}
                if 'name' in parameter.keys():
                    input_param = {parameter['name']: input_param}
            elif type(parameter) == dict and 'name' not in parameter.keys():
                for param in parameter:
                    input_param = self._get_basic_parameter(parameter[param])
            else:
                input_param = {}
        elif re.search('array', parameter['type'], re.I) is not None:
            if 'sub_type' in parameter.keys():
                if re.search('int', parameter['sub_type']):
                    input_param = [random.randint(0, 999999)]
                elif re.search('str', parameter['sub_type']):
                    dt = datetime.now()
                    input_param = [f'TEST_TOKEN_{dt.strftime("%d-%m-%Y_%H-%M-%S")}']
                elif re.search('bool', parameter['sub_type']):
                    input_param = [False]
                elif re.search('object', parameter['sub_type']):
                    input_param = {}
                    for param in parameter['sub_values']:
                        input_param = {**{param['name']: self._get_basic_parameter(param)}, **input_param}
                    input_param = [input_param]
                else:
                    print('hmm')
        elif re.search('int|number|num', parameter['type'], re.I) is not None:
            input_param = random.randint(0, 999999)
        elif re.search('string|str|text', parameter['type'], re.I) is not None:
            dt = datetime.now()
            input_param = f'TEST_TOKEN_{dt.strftime("%d-%m-%Y_%H-%M-%S")}'
        elif re.search('bool', parameter['type'], re.I) is not None:
            input_param = False
        if 'example' in parameter.keys() and parameter['example']:
            input_param = parameter['example']
        return input_param

    def _create_payload(self, req_param):
        header_params   = {}
        path_params     = {}
        body_params     = {}
        query_params    = {}
        for param in req_param:
            if 'type' not in param.keys() and not ('schema' in param.keys() and not 'type' not in param['schema'].keys()):
                print('Error param has no type:' + str(param))
                continue
            elif 'schema' in param.keys() and not 'type' not in param['schema'].keys():
                param['type'] = param['schema']['type']
            input_param = self._get_basic_parameter(param)

            if param['in'] == 'path':
                path_params[param['name']] = input_param
            elif  param['in'] == 'query':
                query_params[param['name']] = input_param
            elif param['in'] == 'body':
                body_params[param['name']]   = input_param
            elif param['in'] == 'header':
                if 'default' in param.keys():
                    header_params[param['name']] = param['default']
                elif 'required' in param.keys() and param['required'] == False:
                    pass
                else:
                    header_params[param['name']] = input_param
        return header_params, path_params, body_params, query_params
