from tensorflow import keras

from modules.data.loader import load_dataset, split
from modules.training import train_model, save_model
from modules.training.optimizers import LearningRateOptimizerFactory
from models import ModelFactory


def train(architecture, dataset_path, lr_scheduler, loss, epochs, batch_size, log_dir, checkpoint_dir, weights_output_path):
    
    dataset, classes = load_dataset(
        data_dir = dataset_path,
        drop_columns = ['position', 'width']
    )

    model = ModelFactory.get(architecture, num_classes = len(classes))

    train_ds, val_ds, test_ds = split(dataset)
    
    scheduler = LearningRateOptimizerFactory.get(lr_scheduler, steps = len(train_ds) * epochs)
    #loss = keras.losses.CategoricalCrossentropy() if len(classes) > 1 else keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(scheduler)

    # to view metrics run the following terminal command:
    # tensorboard --logdir tools/training/logs
    model = train_model(
        model,
        train_ds,
        val_ds,
        batch_size, epochs,
        loss, optimizer,
        metrics=[keras.metrics.Accuracy(), keras.metrics.Recall(), keras.metrics.Precision()],
        log_metrics=True, log_dir = log_dir, save_dir=checkpoint_dir)
    
    save_model(model, weights_output_path)
    

if __name__ == '__main__':
    train("fcnt", 
          'static/time_synth/pattern_dataset_angry_rosalind/angry_rosalind',
          lr_scheduler='stair', 
          loss='categorical_crossentropy',
          epochs=15, 
          batch_size=16, 
          log_dir="static/training/logs/scalars/", 
          checkpoint_dir="static/models/saved/checkpoints/", 
          weights_output_path="static/models/weights")

