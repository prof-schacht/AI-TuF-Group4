import os
import tensorflow as tf
import keras_tuner as kt
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from LoadAndPrepareData import LoadAndPrepareData
import argparse
import logging

# Setup logging
log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "train_run.log"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PowerConsumptionModel:
    """
    Class for training and tuning a time series forecasting model for power consumption data.
    Uses Keras Tuner with Bayesian Optimization for hyperparameter tuning and Weights & Biases for experiment tracking.
    
    Args:
        data_path: Path to the household power consumption data file
        window_size: Number of time steps to use as input
        horizon: Number of time steps to predict
        batch_size: Batch size for training
        project_name: Name for the W&B project
        max_trials: Maximum number of trials for hyperparameter tuning
        executions_per_trial: Number of executions per trial
    """
    
    def __init__(self, 
                data_path: str,
                window_size: int = 24,
                horizon: int = 6,
                batch_size: int = 32,
                project_name: str = "power-consumption-forecasting",
                max_trials: int = 10,
                executions_per_trial: int = 2,
                use_wandb: bool = True,
                test_mode: bool = False,
                resample_rule: str = "h",
                test_subset: int = None):
        
        self.data_path = data_path
        self.window_size = window_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.project_name = project_name
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.use_wandb = use_wandb
        self.test_mode = test_mode
        self.resample_rule = resample_rule
        self.test_subset = test_subset
        
        logger.info("Loading and preparing data...")
        self.data_loader = LoadAndPrepareData(
            filepath=self.data_path,
            window_size=self.window_size,
            horizon=self.horizon,
            batch_size=self.batch_size,
            test_mode=self.test_mode,
            resample_rule=self.resample_rule,
            test_subset=self.test_subset
        )
        
        self.train_ds, self.val_ds, self.test_ds = self.data_loader.get_datasets()
        self.scaler = self.data_loader.get_scaler()
        logger.info("Data loaded. Number of train batches: %s", sum(1 for _ in self.train_ds))
        # Ensure input_shape and output_shape are set
        self.input_shape = None
        self.output_shape = None
        for x, y in self.train_ds.take(1):
            self.input_shape = x.shape[1:]
            self.output_shape = y.shape[1:]
            break
        if self.input_shape is None or self.output_shape is None:
            msg = "Could not determine input/output shape from the training dataset."
            if self.test_mode:
                msg += "\n[TrainModel] In test_mode, try increasing the test subset size or reducing window/horizon/batch_size."
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Input shape: {self.input_shape}, Output shape: {self.output_shape}")
        
        # Initialize W&B if requested
        if self.use_wandb:
            self.init_wandb()
        
    def init_wandb(self):
        """Initialize Weights & Biases for experiment tracking"""
        input_shape_serializable = tuple(self.input_shape) if self.input_shape is not None else None
        output_shape_serializable = tuple(self.output_shape) if self.output_shape is not None else None
        logger.info(f"Initializing W&B with input_shape={input_shape_serializable}, output_shape={output_shape_serializable}")
        wandb.init(project=self.project_name, config={
            "window_size": self.window_size,
            "horizon": self.horizon,
            "batch_size": self.batch_size,
            "input_shape": input_shape_serializable,
            "output_shape": output_shape_serializable
        })
    
    def build_model(self, hp):
        """
        Build model with hyperparameters from Keras Tuner
        
        Args:
            hp: HyperParameters object from Keras Tuner
            
        Returns:
            Compiled Keras model
        """
        # Choose model type
        model_type = hp.Choice('model_type', ['lstm', 'gru', 'bidirectional_lstm'])
        
        # Define model architecture
        model = Sequential()
        
        # First layer
        if model_type == 'lstm':
            model.add(LSTM(
                units=hp.Int('lstm_units_1', min_value=32, max_value=256, step=32),
                activation=hp.Choice('lstm_activation_1', ['relu', 'tanh']),
                return_sequences=hp.Boolean('return_sequences_1'),
                input_shape=self.input_shape
            ))
        elif model_type == 'gru':
            model.add(GRU(
                units=hp.Int('gru_units_1', min_value=32, max_value=256, step=32),
                activation=hp.Choice('gru_activation_1', ['relu', 'tanh']),
                return_sequences=hp.Boolean('return_sequences_1'),
                input_shape=self.input_shape
            ))
        else:  # bidirectional_lstm
            model.add(Bidirectional(
                LSTM(
                    units=hp.Int('bi_lstm_units_1', min_value=32, max_value=256, step=32),
                    activation=hp.Choice('bi_lstm_activation_1', ['relu', 'tanh']),
                    return_sequences=hp.Boolean('bi_return_sequences_1')
                ),
                input_shape=self.input_shape
            ))
        
        # Dropout after first layer
        model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
        
        # Conditional second layer based on return_sequences
        if model_type == 'lstm' and hp.get('return_sequences_1') or \
           model_type == 'gru' and hp.get('return_sequences_1') or \
           model_type == 'bidirectional_lstm' and hp.get('bi_return_sequences_1'):
            
            if model_type == 'lstm':
                model.add(LSTM(
                    units=hp.Int('lstm_units_2', min_value=16, max_value=128, step=16),
                    activation=hp.Choice('lstm_activation_2', ['relu', 'tanh'])
                ))
            elif model_type == 'gru':
                model.add(GRU(
                    units=hp.Int('gru_units_2', min_value=16, max_value=128, step=16),
                    activation=hp.Choice('gru_activation_2', ['relu', 'tanh'])
                ))
            else:  # bidirectional_lstm
                model.add(Bidirectional(
                    LSTM(
                        units=hp.Int('bi_lstm_units_2', min_value=16, max_value=128, step=16),
                        activation=hp.Choice('bi_lstm_activation_2', ['relu', 'tanh'])
                    )
                ))
                
            # Dropout after second layer
            model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
        
        # Dense layers
        model.add(Dense(
            units=hp.Int('dense_units', min_value=16, max_value=128, step=16),
            activation='relu'
        ))
        
        # Output layer - reshape to match the expected output dimensions
        output_size = self.output_shape[0] * self.output_shape[1] if len(self.output_shape) > 1 else self.output_shape[0]
        model.add(Dense(output_size))
        
        # Reshape to match the expected output shape if needed
        if len(self.output_shape) > 1:
            model.add(tf.keras.layers.Reshape(self.output_shape))
        
        # Compile model
        model.compile(
            optimizer=Adam(
                learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
            ),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_callbacks(self, model_dir="models"):
        """
        Create callbacks for training
        
        Args:
            model_dir: Directory to save model checkpoints
            
        Returns:
            List of callbacks
        """
        os.makedirs(model_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Add WandbCallback if W&B is enabled
        if self.use_wandb:
            callbacks.append(WandbCallback())
        
        return callbacks
    
    def tune_hyperparameters(self, epochs=50):
        """
        Tune hyperparameters using Keras Tuner with Bayesian Optimization
        
        Args:
            epochs: Maximum number of epochs for each trial
            
        Returns:
            Best hyperparameters
        """
        tuner = kt.BayesianOptimization(
            self.build_model,
            objective='val_loss',
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,
            directory='tuner_results',
            project_name=self.project_name
        )
        
        # Define early stopping for tuning
        stop_early = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Set up callbacks for tuning
        tune_callbacks = [stop_early]
        if self.use_wandb:
            tune_callbacks.append(WandbCallback())
            
        # Search for best hyperparameters
        tuner.search(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=tune_callbacks
        )
        
        # Get best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Log best hyperparameters to W&B if enabled
        if self.use_wandb:
            wandb.config.update({
                'best_hyperparameters': best_hps.values
            })
        
        return best_hps
    
    def train_best_model(self, best_hps, epochs=100, model_save_path="/Users/sebastiangerz/CascadeProjects/AI-TuF-Group4/src/srv/models/best_model.h5"):
        """
        Train model with best hyperparameters and explicitly save it
        
        Args:
            best_hps: Best hyperparameters from tuning
            epochs: Maximum number of epochs for training
            model_save_path: Path to save the trained model
        Returns:
            Trained model and training history
        """
        # Build model with best hyperparameters
        model = self.build_model(best_hps)
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        # Train model
        history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Explicitly save the model
        import os
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        logger.info(f"Model explicitly saved to: {os.path.abspath(model_save_path)}")
        if self.use_wandb:
            wandb.save(model_save_path)
        
        return model, history
    
    def evaluate_model(self, model):
        """
        Evaluate model on test dataset
        
        Args:
            model: Trained model
            
        Returns:
            Evaluation metrics
        """
        # Evaluate model
        results = model.evaluate(self.test_ds)
        
        # Create metrics dictionary
        metrics = {
            'test_loss': results[0],
            'test_mae': results[1]
        }
        
        # Log evaluation metrics to W&B if enabled
        if self.use_wandb:
            wandb.log(metrics)
        
        return metrics
    
    def run_experiment(self, epochs_tune=50, epochs_train=100):
        """
        Run full experiment: tune hyperparameters, train best model, and evaluate
        
        Args:
            epochs_tune: Maximum number of epochs for tuning
            epochs_train: Maximum number of epochs for training
        Returns:
            Trained model and evaluation metrics
        """
        # Tune hyperparameters
        best_hps = self.tune_hyperparameters(epochs=epochs_tune)
        
        # Train best model
        model, history = self.train_best_model(best_hps, epochs=epochs_train)
        
        # Evaluate model
        metrics = self.evaluate_model(model)
        
        # Finish W&B run if enabled
        if self.use_wandb:
            logger.info("Finishing W&B run...")
            wandb.finish()
        logger.info("Training complete.")
        return model, metrics


if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Define data path - use absolute path
    import os
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/household_power_consumption.txt"))
    
    import argparse
    parser = argparse.ArgumentParser(description="Train power consumption forecasting model")
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with small data subset')
    args = parser.parse_args()

    # Test mode overrides
    if args.test_mode:
        window_size = 4
        horizon = 2
        batch_size = 4
        resample_rule = "min"
        test_subset = 500
        logger.info(f"[TrainModel] test_mode: window_size={window_size}, horizon={horizon}, batch_size={batch_size}, resample_rule={resample_rule}, test_subset={test_subset}")
    else:
        window_size = 24
        horizon = 6
        batch_size = 32
        resample_rule = "h"
        test_subset = None
        logger.info(f"[TrainModel] full mode: window_size={window_size}, horizon={horizon}, batch_size={batch_size}, resample_rule={resample_rule}, test_subset={test_subset}")

    # Create model trainer
    trainer = PowerConsumptionModel(
        data_path=data_path,
        window_size=window_size,
        horizon=horizon,
        batch_size=batch_size,
        project_name="power-consumption-forecasting",
        max_trials=10,   # Number of hyperparameter tuning trials
        executions_per_trial=2,  # Number of executions per trial to reduce variance
        use_wandb=True,  # Enable W&B for full training
        test_mode=args.test_mode,
        resample_rule=resample_rule,
        test_subset=test_subset
    )
    logger.info("Starting full training run...")
    
    # Run experiment
    model, metrics = trainer.run_experiment(
        epochs_tune=3 if args.test_mode else 30,  # Fewer epochs if test_mode
        epochs_train=5 if args.test_mode else 100  # Fewer epochs if test_mode
    )
    
    logger.info(f"Test Loss: {metrics['test_loss']:.4f}")
    logger.info(f"Test MAE: {metrics['test_mae']:.4f}")
    if args.test_mode:
        logger.info("Test run complete. For full training, rerun without --test_mode.")
