import json
import socket
import time


class indiesolver:
    def initialize(self, url, port, token):
        self.url = url
        self.port = port
        self.token = token

    def create_problem(self, problem):
        self.problem = problem
        request = problem
        request["request_type"] = "create_problem"
        return self.send(request)

    def set_problem(self, problem):
        self.problem = problem

    def ask_problem_description(self, name):
        request = {}
        request["problem_name"] = name
        request["request_type"] = "ask_problem_description"
        return self.send(request)

    def ask_new_solutions(self, nsolutions):
        request = {}
        request["problem_name"] = self.problem["problem_name"]
        request["request_type"] = "ask_solutions"
        request["request_argument1"] = "new"
        request["request_argument2"] = nsolutions
        return self.send(request)

    def ask_pending_solutions(self):
        request = {}
        request["problem_name"] = self.problem["problem_name"]
        request["request_type"] = "ask_solutions"
        request["request_argument1"] = "pending"
        return self.send(request)

    def ask_problems(self):
        request = {}
        request["request_type"] = "ask_problems"
        return self.send(request)

    def tell_metrics(self, metrics):
        request = metrics
        request["problem_name"] = self.problem["problem_name"]
        request["request_type"] = "tell_metrics"
        return self.send(request)

    def send(self, request):
        BUFFER_SIZE = 1024
        request["token"] = self.token
        message = json.dumps(request) + '\0'
        while 1:
            try:
                # self.skt.send(message) # Python 2.x
                self.skt.sendto(message.encode(), (self.url, self.port))  # Python 3.x
                message_is_complete = 0
                reply = ''
                while message_is_complete == 0:
                    # reply = reply + self.skt.recv(BUFFER_SIZE) # Python 2.x
                    reply = reply + self.skt.recv(BUFFER_SIZE).decode('utf-8')  # Python 3.x

                    if len(reply) > 0:
                        if reply[len(reply) - 1] == '\0':
                            message_is_complete = 1
                            reply = reply[: len(reply) - 1]

                reply = json.loads(reply)
                if reply["status"] != "success":
                    print(reply)
                return reply
            except Exception as msg:
                print(msg)
                try:
                    self.skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.skt.connect((self.url, self.port))
                except Exception as msg:
                    print(msg)
                    print('reconnect\t')
                    time.sleep(2)

    def disconnect_socket(self):
        try:
            self.skt.close()
        except:
            return

    def __exit__(self):
        self.disconnect_socket()
