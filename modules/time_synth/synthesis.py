from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
from modules import * 
import json
import os
import csv

from modules.time_synth.timeseries import Timeseries, TimeseriesBoundingbox


def run_synthesis(configuration_path, output_directory='./out'):
    with open(configuration_path, 'r') as config_json:
        configuration = json.load(config_json)

    # This can be smarter
    patterns = configuration.get("patterns") 

    pattern_size = configuration.get("pattern_size", 100)
    randomize_pattern = configuration.get('randomize_pattern', (4, 6))
    point_noise_level = configuration.get("point_noise_level", 0.75)
    pattern_noise_level = configuration.get("pattern_noise_level", 0.1)
    max_offset_level = configuration.get("max_offset_level", 1)

    padding_side = configuration.get("padding_side", 'left')
    padding_point_noise = configuration.get("padding_point_noise", 0.)
    padding_noise_level = configuration.get("padding_noise_level", 0.)


    samples = []
    labels = []

    for pattern_config in patterns:
        name = pattern_config.get("name")
        points = pattern_config.get("points")
        num_samples = pattern_config.get("num_samples", 1000)
        
        for sample in tqdm(range(num_samples)):
            bounding_box = TimeseriesBoundingbox()
            pattern = Timeseries(points)\
                        .add_noise(point_noise_level)\
                        .randomly_increase_precision(randomize_pattern)\
                        .increase_precision_quadratic(2)\
                        .add_noise(pattern_noise_level if name!='BASE' else padding_noise_level)\
                        .downsample_to(pattern_size)\
                        .offset(max_offset_level)\
                        .random_choice(pattern_size // 5, pattern_size)\
                        .pad_signal(point_noise_level=padding_point_noise,
                                    noise_level = padding_noise_level,
                                    max_size = pattern_size, 
                                    side=padding_side, 
                                    source_bb=bounding_box)\
                        .offset(max_offset_level)\
                        .normalize()
            
            samples.append(pattern)
            labels.append({"name":name, "position":bounding_box.position, "width": bounding_box.width})
    

    with open(os.path.join(output_directory, "dataset.csv"), 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(samples)

    with open(os.path.join(output_directory, "labels.csv"), 'w') as f:
        csv_file = csv.DictWriter(f, 
                        fieldnames=labels[0].keys())
        csv_file.writeheader()
        csv_file.writerows(labels)
    
    


if __name__ == '__main__':
    def _get_parameters():
        parser = ArgumentParser()
        parser.add_argument('-c', '--configuration_path',help="Path to the configuration json that describes the Dataset", 
                            default = "./configuration_example.json")
        parser.add_argument('-o', '--output_directory', default='../../static/time_synth/out',
                            help="Directory to save the created dataset")
        

        args = parser.parse_args()
        return args

    args = _get_parameters()
    args_dict = vars(args)
    run_synthesis(**args_dict)