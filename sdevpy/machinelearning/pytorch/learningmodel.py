""" Wrapper class for PyTorch learning models, including scalers, and simplifying
    evaluation, history tracking, exporting to/importing from files, etc. """
import logging
from pathlib import Path
import numpy.typing as npt
import torch
from torch.utils.data import TensorDataset, DataLoader
# from torch.optim import Optimizer
import joblib
# from sdevpy.utilities import jsonmanager as jsm
from sdevpy.machinelearning.learningmodel import LearningModel, scaler_files
from sdevpy.machinelearning.pytorch import learningschedules as lrmod
log = logging.getLogger(__name__)


class TorchLearningModel(LearningModel):
    """ PyTorch subclass of LearningModel """
    def __init__(self, model, is_scaled=False, x_scaler=None, y_scaler=None, device=None):
        super().__init__(model, is_scaled, x_scaler, y_scaler)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = self.model.to(device)
        self.loss = None
        self.optimizer = None
        self.scheduler = None
        self.epoch_sampling, self.x_test, self.y_test = None, None, None

    def set_loss(self, loss) -> None:
        """ Set loss function """
        self.loss = loss

    def set_optimizer(self, optimizer_type: str, init_lr: float, **kwargs) -> None:
        """ Set optimizer. Also sets the default learning rate scheduler to be constant at
            init_lr. To set a different LR scheduler, call set_lr_scheduler() after setting the
            optimizer
        """
        match optimizer_type.lower():
            case 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)
            case _:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Set constant scheduler by default
        self.scheduler = lrmod.create_scheduler("constant", self.optimizer)

    def set_lr_scheduler(self, name: str, **kwargs) -> None:
        """ Set learning rate scheduler. If not called, the learning rate with default to constant
            with the initial LR value set during set_optimizer().
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Set the optimizer before setting the LR scheduler")
        self.scheduler = lrmod.create_scheduler(name, self.optimizer, **kwargs)

    def train_on_scaled(self, x_scaled: npt.ArrayLike, y_scaled: npt.ArrayLike, epochs: int, batch_size: int,
                        shuffle: bool) -> None:
        """ Training on scaled data (both x and y) """
        if self.loss is None:
            raise ValueError("Training aborted: loss function not set")

        if self.optimizer is None:
            raise ValueError("Training aborted: optimizer not set")

        if self.scheduler is None:
            raise ValueError("Training aborted: scheduler not set")

        # Convert to tensors
        x_t = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

        # DataLoader
        loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)

        # Training history
        self.hist_epochs, self.hist_losses, self.hist_lr = [], [], []
        self.test_epochs, self.test_losses = [], []

        log.info("<><><><><><><><> TRAINING START <><><><><><><><>")
        log.info(f"Epochs: {epochs}")
        log.info(f"Batch size: {batch_size:,}")
        log.info(f"Training set size: {len(x_t):,}")
        log.info("<><><><><><><><><><><><><><><><><><><><><><><><>")

        for epoch in range(epochs):
            log.info(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad() # Reset the gradients
                pred = self.model(batch_x) # Feed-forward
                loss = self.loss(pred, batch_y) # Calculate loss
                loss.backward() # Propagate backwards
                self.optimizer.step() # Modify weights
                self.scheduler.step() # Evolve learning rate
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            current_lr = self.scheduler.get_last_lr()[0]
            log.info(f"Loss: {avg_loss:.2f}, LR: {current_lr:.6f}")
            self.hist_epochs.append(epoch)
            self.hist_losses.append(avg_loss)
            self.hist_lr.append(current_lr)

            # Run sample test (if set)
            self.sample_test(epoch)

            log.info("<><><><><><><><><><><><><><><><>")

    def sample_test(self, epoch: int) -> None:
        """ Estimate the model on test set """
        if self.epoch_sampling is not None:
            if epoch == 0: # Scale the data only once
                self.x_test = torch.tensor(self.x_scaler.transform(self.x_test), dtype=torch.float32).to(self.device)
                self.y_test = torch.tensor(self.y_scaler.transform(self.y_test), dtype=torch.float32).to(self.device)

            if epoch == 0 or (epoch + 1) % self.epoch_sampling == 0:
                self.model.eval()
                with torch.no_grad():
                    y_pred_scaled = self.model(self.x_test)
                    test_loss = self.loss(y_pred_scaled, self.y_test).item()
                self.test_epochs.append(epoch)
                self.test_losses.append(test_loss)
                log.info(f"Test loss: {test_loss:.2f}")

    def set_sample_testing(self, epoch_sampling: int=None,
                           x_test: npt.ArrayLike=None, y_test: npt.ArrayLike=None) -> None:
        """ Set sample testing to run every epoch_sampling epochs
            Args:
                - epoch_sampling: number of epoch length at which we sample. Defaults to None meaning no sampling.
                - x_test, y_test: data on which to test the model
        """
        self.epoch_sampling = epoch_sampling
        if self.epoch_sampling is not None:
            self.x_test, self.y_test = x_test, y_test
            # Check test data is provided
            if x_test is None or y_test is None:
                raise ValueError("Invalid test data provided")

    def predict_on_scaled(self, x_scaled: npt.ArrayLike) -> npt.ArrayLike:
        """ Predict (on scaled x, outputting scaled y) """
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
            y_t = self.model(x_t)

        return y_t.cpu().numpy()

    def save(self, path: Path):
        """ Save model and its scalers to files """
        pass


def load_model(path: Path) -> TorchLearningModel:
    """ Load PyTorch learning model from files """
    if not path.exists():
        raise RuntimeError(f"Model folder does not exist: {path}")

    model_file = path / "torchmodel.pt"
    # torch_model = tf.keras.models.load_model(model_file)
    torch_model = 'todo' + model_file

    x_scaler_file, y_scaler_file = scaler_files(path)
    if x_scaler_file.exists() and y_scaler_file.exists():
        x_scaler = joblib.load(x_scaler_file)
        y_scaler = joblib.load(y_scaler_file)
        model = TorchLearningModel(torch_model, is_scaled=True, x_scaler=x_scaler, y_scaler=y_scaler)
    else:
        model = TorchLearningModel(torch_model)

    # config_file = path / 'config.json'
    # if config_file.exists():
    #     config_data = jsm.deserialize(config_file)
    #     model.topology_ = config_data['topology']

    return model
