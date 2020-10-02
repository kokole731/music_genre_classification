import argparse

def hello():
    print('hellp')
class HParam():

    def __init__(self, ):

        self.dataset_origin_path = '../dataset/src_data' 
        self.dataset_target_path = '../dataset/target_data'
        self.feature_path = '../dataset/feature'

        self.genres = ["Foxtrot", "Pasodoble", "Rumba", "Waltz", "Viennesewaltz", "Chacha", "Jive", "Tango", "Samba", "Quickstep"]


        # feature parameter
        self.sample_rate = 22050
        self.fft_size = 1024
        self.win_length = 1024
        self.hop_length = 512
        self.feature_size = 1024
        self.num_mels = 128
        self.duration = 1024 
        
        # training paramter
        self.device = 0  # 0: cpu, 1: gpu
        self.batch_size = 128
        self.num_epochs = 26
        self.learning_rate = 0.001
        self.stopping_rate = 1e-5
        self.weight_decay = 1e-6
        self.momentum = 0.9
        self.factor = 0.2
        self.patience = 5


hps = HParam()


    