import sys
import yaml
import json
from collections import ChainMap
from pre_processing.request import Request
import re
from openapi_parser import parse
from openapi_parser.parser import Property, Object
from pre_processing.request import Request

def create_requests(filename):
    if '.yaml' in filename:
        api_spec = parse(filename, False)
    elif '.json' in filename:
        api_spec = parse(filename, False)
    else:
        print(filename)
        print('Could not recognise file format')
        sys.exit(2)
    request_sequence = []
    host = api_spec.servers[0].url
    for paths in api_spec.paths:
        req_path = paths.url
        for request in paths.operations:
            request_type = request.method.value

            parameters = []
            if request.request_body is not None:
                for body_content in request.request_body.content:
                    if body_content.type.value == 'application/json':
                        if type(body_content.schema) == Property:
                            parameters.append(get_param_details(param))
                        if type(body_content.schema) != Object:
                            if type(body_content.schema.items) == Object:
                                for param in body_content.schema.items.properties:
                                    parameters.append(get_param_details(param))
                            else:
                                print('hard to parse')

                        else:
                            for param in body_content.schema.properties:
                                parameters.append(get_param_details(param))

            if len(request.parameters) > 0:
                for param in request.parameters:
                    parameters.append(get_param_details(param))

            if request.summary is not None and len(request.summary) > 0:
                request_sequence.append(Request('', req_path, request_type, parameters, host, request.summary))
            else:
                request_sequence.append(Request('', req_path, request_type, parameters, host, ''))

    return request_sequence





def get_param_details(param):
    found_param = {}
    if param.schema.type.value == 'object':
        sub_params = []
        for sub_param in param.schema.properties:
            sub_params.append(get_param_details(sub_param))
        found_param['sub_values'] = sub_params
        found_param['name'] = param.name
        found_param['type'] = param.schema.type.value

        try:
            found_param['in'] = param.location.value
        except:
            found_param['in'] = 'body'
        try:
            if param.required is not None:
                 found_param['example'] = param.required
        except:
            pass
    elif param.schema.type.value == 'array':
        found_param['sub_type'] = param.schema.items.type.value
        found_param['name'] = param.name
        found_param['type'] = param.schema.type.value

        try:
            found_param['in'] = param.location.value
        except:
            found_param['in'] = 'body'
        try:
            if param.required is not None:
                 found_param['example'] = param.required
        except:
            pass
    else:
        found_param['name'] = param.name
        try:
            found_param['in'] = param.location.value
        except:
            found_param['in'] = 'body'
        found_param['type'] = param.schema.type.value
        if param.schema.example is not None:
             found_param['example'] = param.schema.example
        try:
            if param.required is not None:
                 found_param['required'] = param.required
        except:
            pass

        if param.schema.default is not None:
             found_param['example'] = param.schema.default
    return found_param

def custom_parse(filename):
    if '.yaml' in filename:
        import os
        
        with open(filename, 'r') as file:
            try:
                api_spec = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
            #print(api_spec)
    elif '.json' in filename:
        with open(filename, 'r') as file:
            api_spec = json.load(file)
    else:
        print(filename)
        print('Could not recognise file format')
        sys.exit(2)

    

    request_sequence = []
    for req_path in api_spec['paths']:
        if 'order/{orderId}' in req_path:
            print('jn')
        for request_type in api_spec['paths'][req_path]:
            parameters = []
            if 'parameters' in api_spec['paths'][req_path].keys():
                parameters += api_spec['paths'][req_path]['parameters']
            if 'parameters' in api_spec['paths'][req_path][request_type]:
                parameters += api_spec['paths'][req_path][request_type]['parameters']
            if next(gen_dict_extract('properties', api_spec['paths'][req_path]), None) is not None:

                is_other_request_type = False
                alt_req_params = {}
                if next(gen_dict_extract('properties', next(gen_dict_extract('responses', api_spec['paths'][req_path]), None)), None) is not None:
                    all_params = return_properties_as_parameters(next(gen_dict_extract('properties',next(gen_dict_extract(request_type, api_spec['paths'][req_path]), None)), None))
                    response_params = return_properties_as_parameters(next(gen_dict_extract('properties', next(gen_dict_extract('responses', next(gen_dict_extract(request_type, api_spec['paths'][req_path]), None)), None)), None))
                    request_params = [json.loads(i)  for i in list(set([json.dumps(param) for param in all_params]) - set([json.dumps(param) for param in response_params]) - set([json.dumps(param) for param in alt_req_params]) )]
                    parameters += request_params
                else:
                    parameters += return_properties_as_parameters(next(gen_dict_extract('properties', api_spec['paths'][req_path])))
            elif next(gen_dict_extract('requestBody', api_spec['paths'][req_path]), None) is not None:
                alt_req_params = {}
                if next(gen_dict_extract('requestBody',
                                         next(gen_dict_extract('responses', api_spec['paths'][req_path]), None)),
                        None) is not None:
                    all_params = return_properties_as_parameters(next(gen_dict_extract('requestBody', next(
                        gen_dict_extract(request_type, api_spec['paths'][req_path]), None)), None))
                    response_params = return_properties_as_parameters(next(gen_dict_extract('requestBody', next(
                        gen_dict_extract('responses',
                                         next(gen_dict_extract(request_type, api_spec['paths'][req_path]), None)),
                        None)), None))
                    request_params = [json.loads(i) for i in list(
                        set([json.dumps(param) for param in all_params]) - set(
                            [json.dumps(param) for param in response_params]) - set(
                            [json.dumps(param) for param in alt_req_params]))]
                    parameters += request_params
                else:
                    parameters += return_requestBody_as_parameters(
                        next(gen_dict_extract('requestBody', api_spec['paths'][req_path]), None))

            params_to_remove = []
            params_to_add = []
            
            for param in parameters:
                if next(gen_dict_extract('$ref',param), None) is not None:
                    if 'in' in param.keys():
                        new_params = return_schema(next(gen_dict_extract('$ref', param)), api_spec, param, param['in'])
                       
                    else:
                        new_params = return_schema(next(gen_dict_extract('$ref', param)), api_spec, param)
                    if len(new_params) == 1:
                        parameters[parameters.index(param)] = {**param, **new_params[0]}
                    else:
                        parameters[parameters.index(param)] = {**param, **new_params[0]}
                        for new_param in new_params[1:]:
                            params_to_add.append({**param, **new_param})

                if 'in' in param.keys() and param['in'] == None:
                    print('h')
            for param in params_to_add:
                if re.search('header', param['in'], re.I) is not None:
                    continue
                parameters.append(param)
            for param in params_to_remove:
                parameters.remove(param)
            if request_type != 'parameters':
                host = api_spec['host'] if 'host' in api_spec.keys() else api_spec['servers'][0]['url']
                base_path = api_spec['basePath'] if 'basePath' in api_spec.keys() else ''
                if 'summary' in  api_spec['paths'][req_path][request_type]:
                    summary = api_spec['paths'][req_path][request_type]['summary']
                else:
                    summary = ''
                request_sequence.append(Request(base_path, req_path, request_type, parameters, host, summary))
    return request_sequence


def gen_dict_extract(key, var):
    if hasattr(var,'items'):
        for k, v in iter(var.items()):
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result


def parse_properties(dic):
    re ={}
    for k,v in dic.items():
        if isinstance(v, dict):
            re.update({k:parse_properties(v)})
        else:
            re.update({k: v})
    return re


def return_properties_as_parameters(properties):
    parameters = []
    for property in properties.keys():
        if 'properties' in property:
            parameters += [return_properties_as_parameters(properties[property])]
        elif isinstance(properties[property], dict):
            if next(gen_dict_extract('properties', properties[property]), None) is not None:
                parameters += [{'name': property, 'in': 'body', 'type': properties[property]['type'], 'sub_values': return_properties_as_parameters(next(gen_dict_extract('properties', properties[property])))}]
            else:
                new_param = {'name':property, 'in': 'body'}
                new_param.update(parse_properties(properties[property]))
                parameters += [new_param]
        elif isinstance(properties[property], list):
            print('panic')
    return parameters

def return_requestBody_as_parameters(properties):
    parameters = []
    for property in properties.keys():
        if 'requestBody' in property:
            parameters += [return_properties_as_parameters(properties[property])]
        elif isinstance(properties[property], dict):
            if next(gen_dict_extract('requestBody', properties[property]), None) is not None:
                parameters += [{property: return_properties_as_parameters(next(gen_dict_extract('requestBody', properties[property])))}]
            else:
                new_param = {'name':property, 'in': 'body'}
                new_param.update(parse_properties(properties[property]))
                parameters += [new_param]
    return parameters

def find_node(schema, path):
    new_path = ''
    new_path_without_slashes = ''
    path_to_find = path
    path_left_to_find = path
    possible_schemas = {}
    for sub_path in path_to_find:
        print(new_path)
        # path with slashes
        if new_path in schema.keys():
            #schema = schema[new_path]
            possible_schemas[new_path] = [schema[new_path], path_left_to_find]
            path_left_to_find = path_left_to_find[1:]
            #new_path = '/'.join([new_path, sub_path])
        # account for no slashes
        elif new_path_without_slashes in schema.keys():
            possible_schemas[new_path] = [schema[new_path_without_slashes], path_left_to_find]
            path_left_to_find = path_left_to_find[1:]
        # account for no beginning slash
        elif new_path[1:] in schema.keys():
            #schema = schema[new_path]
            possible_schemas[new_path] = [schema[new_path[1:]], path_left_to_find]
            path_left_to_find = path_left_to_find[1:]
            #new_path = '/'.join([new_path, sub_path])
        elif sub_path in schema.keys():
            path_left_to_find = path_left_to_find[1:]
            possible_schemas[sub_path] = [schema[sub_path], path_left_to_find]
        else:
            path_left_to_find = path_left_to_find[1:]
        new_path = '/'.join([new_path, sub_path])
        new_path_without_slashes = ''.join([new_path_without_slashes, sub_path])

    schema, path_left = possible_schemas[max(possible_schemas, key=len)]
    return schema, path_left

def return_schema(schema_path, api_spec, param, request_loc=None):
    path = schema_path.split('/')
    path.remove('#')
    schema = api_spec
    try:
        path_left = path
        for item in path:
            schema = schema[item]
            path_left = path_left [1:]
    except KeyError:
        while len(path_left) > 0:

            schema, path_left = find_node(schema, path_left)

        return schema

    schema_properties = []
    if 'properties' in schema.keys():
        sub_vals = []
        for property in schema['properties'].keys():
            if request_loc is None:
                request_loc = next(gen_dict_extract('in',  schema['properties'][property]), 'body')
            property_dict = {**{'name': property}, **schema['properties'][property], **{'in': request_loc}}
            if 'schema' in  schema['properties'][property].keys():
                property_dict['type'] = schema['properties'][property]['schema']['type']
            elif 'type' in schema.keys():
                property_dict['type'] = schema['type']
            if 'example' in property_dict.keys():
                property_dict['default'] = property_dict['example']
            sub_vals.append(property_dict)
        if 'type' not in schema and 'type' in param:
            schema['type'] = param['type']
        schema_properties.append({'sub_values':sub_vals, 'type':schema['type'], 'in':'body', 'name':item})
    else:
        if request_loc is None:
            request_loc = schema['in']
        property_dict = {**schema, **{'in': request_loc}}
        if 'name' in schema.keys():
            property_dict['name'] = schema['name']
        if 'schema' in schema.keys() and 'type' not in schema.keys():
            property_dict['type'] = schema['schema']['type']
        elif 'type' in schema.keys():
            property_dict['type'] = schema['type']
            if schema['type'] == 'array':
                schema['sub_type']:  schema['items']['type']
        if 'example' in property_dict.keys():
            property_dict['default'] = property_dict['example']
        schema_properties.append(property_dict)
    return schema_properties


def strip_to_param_requests(requests):
    stripped_requests = []
    for request in requests:
        if len(request.parameters) > 0:
            stripped_requests.append(request)

    return stripped_requests
def remove_delete_requests(requests):
    stripped_requests = []
    for request in requests:
        if request.type != 'delete':
            stripped_requests.append(request)
    return stripped_requests

