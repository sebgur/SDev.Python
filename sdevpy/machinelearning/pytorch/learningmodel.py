""" Wrapper class for PyTorch learning models, including scalers, and simplifying
    evaluation, history tracking, exporting to/importing from files, etc. """
import logging
from pathlib import Path
import numpy.typing as npt
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib
# from sdevpy.utilities import jsonmanager as jsm
from sdevpy.machinelearning.learningmodel import LearningModel, scaler_files
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

    def set_loss(self, loss) -> None:
        """ Set loss function """
        self.loss = loss

    def set_optimizer(self, optimizer_type, lr_scheduler) -> None:
        """ Set optimizer """
        match optimizer_type.lower():
            case 'adam':
                init_lr = lr_scheduler.init_lr
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)
            case _:
                raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_scheduler.step_function)

    def train_raw(self, x_scaled: npt.ArrayLike, y_scaled: npt.ArrayLike, epochs, batch_size, shuffle,
                  callbacks=None):
        """ Training (scaling already done) """
        if self.loss is None:
            raise ValueError("Training aborted: loss function not set")

        if self.optimizer is None:
            raise ValueError("Training aborted: optimizer not set")

        # Convert to tensors
        x_t = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_scaled, dtype=torch.float32).to(self.device)

        # DataLoader
        loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True)

        # Training history
        self.hist_epochs, self.hist_losses, self.hist_lr, sampled_epochs, test_losses = [], [], [], [], []

        log.info("<><><><><><><><> TRAINING START <><><><><><><><>")
        log.info(f"Epochs: {epochs}")
        log.info(f"Batch size: {batch_size:,}")
        log.info(f"Training set size: {len(x_t):,}")
        log.info("<><><><><><><><><><><><><><><><><><><><><><><><>")

        for epoch in range(epochs):
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
            self.hist_epochs.append(epoch)
            self.hist_losses.append(avg_loss)
            self.hist_lr.append(current_lr)

            if callbacks is not None:
                for callback in callbacks:
                    callback.call(epoch, avg_loss, current_lr)

    def predict_raw(self, x_scaled: npt.ArrayLike) -> npt.ArrayLike:
        """ Predict (x-scaling already done, y-scaling not done) """
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
