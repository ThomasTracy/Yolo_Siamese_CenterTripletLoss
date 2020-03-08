import os
import json
import time
import logging


class Params():
    """
    Load hyperparameters from json file
    param = Params(json_path)
    param.batch_size --> 64
    """
    def __init__(self, json_path):
        self._update(json_path)

    def _update(self, json_path):

        # Load parameters from json file
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):

        # Save parameters into json file
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def dict(self):
        # Allow visit parameters through dict way
        # param.dict["batch_size"]
        return self.__dict__

def set_logger(log_path):

    log_name = time.strftime('%Y%m%d_%H-%M-%S', time.localtime(time.time()))
    log_name = 'logging_' + log_name
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        # Logging to file
        file_handler = logging.FileHandler(os.path.join(log_path, log_name))
        file_handler. setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


if __name__ == '__main__':
    json_file = '../model/parameters.json'
    params = Params(json_file)
    print(params.learning_rate)

    set_logger('../logging')
    logging.info('now haha')
    test()