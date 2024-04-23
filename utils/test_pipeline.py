import os
import ray
import yaml
import glob
from collections import namedtuple
import datetime
import shutil
from ray import tune, air 
from utils.config.create_algorithm import create_algorithm
from utils.config.create_config import create_config as crt_cfg

def run_test_pipeline():
        
    ray.init(local_mode=True,
        num_cpus=10,
        num_gpus=0,
        _temp_dir=os.path.dirname(os.getcwd()) + '/' + 'ray_logs',
        include_dashboard=False
        )
    files = glob.glob("configs/*/*.yml")

    for file in files:
        # Load the config file
        with open(file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        config = namedtuple("ObjectName", data.keys())(*data.values())
        path = None 

        algorithm = create_algorithm(config)
        param_space = crt_cfg(config, config.env_config)
            
        # Create the name tags for the trials and log directories
        name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config.type + '_' + config.alg 
        ray_path = os.getcwd() + '/logs/test_pipeline'
        path = ray_path + '/' + name

        # Copy the config files into the ray-run folder
        os.makedirs(os.path.dirname(path + '/'), exist_ok=True)
        shutil.copy(file, path + '/alg_config.yml')
        
        def trial_name_creator(trial):
            return trial.__str__() + '_' + trial.experiment_tag + ','
        
        if config.type == 'ES':
            algorithm = tune.with_resources(algorithm, tune.PlacementGroupFactory([
                            {'CPU': 1}] + 
                            [{'CPU': config.algorithm_config['num_cpus_per_worker']}] * (config.algorithm_config['num_workers']+1)))
                
        tuner = tune.Tuner(
            algorithm,
            tune_config=tune.TuneConfig(num_samples=config.ray_num_trial_samples,
                                        trial_dirname_creator=trial_name_creator),
            param_space=param_space,
            run_config=air.RunConfig(stop={"training_iteration": 1},
                                    local_dir=ray_path,
                                    name=name,
                                    checkpoint_config=air.CheckpointConfig(checkpoint_frequency=config.checkpoint_freq, 
                                                                            checkpoint_at_end=config.checkpoint_at_end)), 
        )
        
        tuner.fit()
        ray.shutdown()

    print('Everything works!')
    exit()