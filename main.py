import argparse

from train import train
from modules.time_synth import run_synthesis
from models import model_init_functions


def invalid_subcommand(*args, **kwargs):
    raise ValueError("Invalid subcommand!")


def test(*args, **kwargs):
    print("Testing neural network...")
    # TODO Add testing code here
    

def main():
    parser = argparse.ArgumentParser(description="Neural Network Train and Test")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="Train a neural network")
    train_parser.add_argument("-a", "--architecture", choices=model_init_functions.keys(), default='fcnt',
                         help="Choose a neural network architecture from the list")
    train_parser.add_argument("-d", "--dataset_path", help="Path to dataset containing dataset.csv and labels.csv files")
    train_parser.add_argument("--lr_scheduler", choices=["cosine", "stair"], default="stair",
                         help="Learning rate scheduler")
    train_parser.add_argument("--loss", default='categorical_crossentropy', help="Loss function. See tf.keras.losses")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--log_dir", help="Logging directory", default='static/logs/')
    train_parser.add_argument("--checkpoint_dir", help="Checkpoint directory", default='static/models/saved/checkpoints')
    train_parser.add_argument("--weights_output_path", help="File to output trained weights", default='static/models/')
    # ###

    test_parser = subparsers.add_parser("test", help="Test a trained neural network")
    test_parser.add_argument("--model_path", required=True, help="Path to the trained model")
    test_parser.add_argument("--test_data", required=True, help="Path to test data")
    # ###

    synthesis_parser = subparsers.add_parser("timesynth", help="Run timeseries synthesis")
    synthesis_parser.add_argument('-c', '--configuration_path',help="Path to the configuration json that describes the Dataset", 
                            default = "./tools/time_synth/configuration_example.json")
    synthesis_parser.add_argument('-o', '--output_directory', default='./static/time_synth/out',
                            help="Directory to save the created dataset")
        
    # ###

    args = parser.parse_args()
    kwargs = vars(args)
    subcommand = kwargs.pop('subcommand')
    
    commands = { "train": train, 
                 "test": test,
                 "timesynth": run_synthesis }
    
    commands.get(subcommand)(**kwargs)


if __name__ == "__main__":
    main()