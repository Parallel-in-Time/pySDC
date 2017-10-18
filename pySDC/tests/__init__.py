import os

def setUp(self):
    if not os.path.exists('data'):
        os.makedirs('data')
