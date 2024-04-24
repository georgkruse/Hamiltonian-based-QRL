import argparse
import datetime
import os
from collections import namedtuple
import numpy as np
import ray
import torch
import yaml
import shutil
from ray import tune, air 
from utils.config.create_algorithm import create_algorithm
from utils.config.create_config import create_config as crt_cfg
from utils.plotting import plotting
from utils.algorithm_benchmarking import algorithm_benchmarking
from utils.landscape_plotting import landscape_plotting
from utils.variance_calculation import variance_calculation
from utils.test_pipeline import run_test_pipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This is the runfile for the open source baseline repo.")
    parser.add_argument("--alg", default= "configs/qppo/qppo_pendulum.yml", 
                        metavar="FILE", help="path to alg config file", type=str)
    parser.add_argument("--test", default='None', type=str)
    args = parser.parse_args()
    
    # Run test
    if args.test == '0':
        run_test_pipeline()
    # Load the config file
    with open(args.alg) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    config = namedtuple("ObjectName", data.keys())(*data.values())
    path = None 

    if 'qrl_training' in config.run_sections:
        ray.init(local_mode=config.ray_local_mode,
                num_cpus=config.total_num_cpus,
                num_gpus=config.total_num_gpus,
                _temp_dir=os.path.dirname(os.getcwd()) + '/' + 'ray_logs',
                include_dashboard=False
                )
        
        if config.constant_seed:
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.set_default_dtype(torch.float32)

        algorithm = create_algorithm(config)
        param_space = crt_cfg(config, config.env_config)
            
        # Create the name tags for the trials and log directories
        name = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + '_' + config.type + '_' + config.alg 
        ray_path = os.getcwd() + '/' + config.ray_logging_path 
        path = ray_path + '/' + name

        # Copy the config files into the ray-run folder
        os.makedirs(os.path.dirname(path + '/'), exist_ok=True)
        shutil.copy(args.alg, path + '/alg_config.yml')
        
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
            run_config=air.RunConfig(stop={"training_iteration": config.training_iterations},
                                    local_dir=ray_path,
                                    name=name,
                                    checkpoint_config=air.CheckpointConfig(checkpoint_frequency=config.checkpoint_freq, 
                                                                            checkpoint_at_end=config.checkpoint_at_end)), 
        )
        
        tuner.fit()
        ray.shutdown()

    if 'algorithm_benchmarking' in config.run_sections:
        algorithm_benchmarking(path=args.alg)

    if 'plotting' in config.run_sections:
        plotting(path=path, config=config)

    if 'landscape_plotting' in config.run_sections:
        landscape_plotting(path=args.alg)

    if 'variance_calculation' in config.run_sections:
        variance_calculation(path=args.alg)