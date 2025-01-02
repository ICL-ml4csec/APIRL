
class Request:
    def __init__(self, base_path, relative_path, request_type, parameters, host, summary):
        if 'http' not in host:
            host = 'http://'+host
        self.path       = host + base_path + relative_path
        self.type       = request_type.lower()
        self.parameters = parameters
        self.summary = summary

