""" Wrapper class for PyTorch learning models, including scalers, and simplifying
    evaluation, history tracking, exporting to/importing from files, etc. """
import logging
from pathlib import Path
import numpy as np
import numpy.typing as npt
from abc import abstractmethod
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib
import random
from sdevpy.machinelearning.learningmodel import LearningModel, scaler_files, MlpTopology
from sdevpy.machinelearning.pytorch import learningschedules as lrmod
from sdevpy.machinelearning.pytorch.topology import compose_mlp
log = logging.getLogger(__name__)


class TorchLearningModel(LearningModel):
    """ PyTorch subclass of LearningModel """
    def __init__(self, torch_model, device=None):
        super().__init__(torch_model)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.base_model = self.base_model.to(device)
        self.loss = None
        self.scheduler = None
        self.epoch_sampling, self.x_test, self.y_test = None, None, None

    def set_loss(self, loss) -> None:
        """ Set loss function """
        self.loss = loss

    def set_optimizer(self, optimizer_type: str, init_lr: float, **kwargs) -> None:
        """ Set optimizer. Also sets the default learning rate scheduler to be constant at
            init_lr. To set a different LR scheduler, call set_lr_scheduler() after setting the
            optimizer.
        """
        match optimizer_type.lower():
            case 'adam':
                self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=init_lr)
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
            self.base_model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad() # Reset the gradients
                pred = self.base_model(batch_x) # Feed-forward
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

        log.info("<><><><><><><><> TRAINING END <><><><><><><><>")

    def sample_test(self, epoch: int) -> None:
        """ Estimate the model on test set """
        if self.epoch_sampling is not None:
            if epoch == 0: # Scale the data only once
                self.x_test = torch.tensor(self.x_scaler.transform(self.x_test), dtype=torch.float32).to(self.device)
                self.y_test = torch.tensor(self.y_scaler.transform(self.y_test), dtype=torch.float32).to(self.device)

            if epoch == 0 or (epoch + 1) % self.epoch_sampling == 0:
                self.base_model.eval()
                with torch.no_grad():
                    y_pred_scaled = self.base_model(self.x_test)
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
        self.base_model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)
            y_t = self.base_model(x_t)

        return y_t.cpu().numpy()

    def save(self, path: Path):
        """ Save model and its scalers to files in the provided path """
        path.mkdir(parents=True, exist_ok=True)

        # Save topology
        topology_file = path / "topology.json"
        self.save_topology(topology_file)

        # Save weights
        weight_file = path / "weights.pt"
        torch.save(self.base_model.state_dict(), weight_file)

        # Save scalers
        self.save_scalers(path)

    @abstractmethod
    def save_topology(self, file: Path) -> None:
        """ Save topology to file """
        pass

    def diagnose_gradients(self, x_set: npt.ArrayLike, y_set: npt.ArrayLike,
                           batch_size: int = 256, use_dropout=True) -> dict:
        """ Run a single forward/backward pass on a batch and report gradient stats per layer """
        if self.loss is None or self.optimizer is None:
            raise ValueError("Set loss and optimizer before running gradient diagnostics")

        # Trigger the computation of the gradients
        if use_dropout:
            # With dropout effect, mimicking training. But has dropout noise so less reproducible.
            self.base_model.train()
        else:
            # No dropout effect, reproducible results. But not exactly what happens during training.
            self.base_model.eval()

        x_scaled = self.x_scaler.transform(x_set)
        y_scaled = self.y_scaler.transform(y_set)
        x_t = torch.tensor(x_scaled[:batch_size], dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_scaled[:batch_size], dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad() # To reset gradient accumulation
        pred = self.base_model(x_t)
        loss = self.loss(pred, y_t)
        loss.backward()

        return self._compute_gradient_stats()

    def _compute_gradient_stats(self) -> dict:
        """ Per-parameter gradient statistics after a backward pass has populated .grad """
        stats = {}
        for name, param in self.base_model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            stats[name] = {
                'mean_abs': grad.abs().mean().item(),
                'std': grad.std(unbiased=False).item(),
                'norm': grad.norm().item(),
                'max_abs': grad.abs().max().item(),
            }
        return stats


class TorchMultiLayerPerceptron(TorchLearningModel):
    """ Multi-layer perceptron wrapper for easier input """
    def __init__(self, topology: MlpTopology, device=None):
        """ Args:
                - hidden_activations: a list of strings representing the activation functions for each
                                      hidden layer
                - n_neurons: number of neurons per hidden layer
        """
        torch_model = compose_mlp(topology)
        super().__init__(torch_model, device)
        self.topology = topology

    def save_topology(self, file: Path) -> None:
        self.topology.to_json(file)


def load_model(path: Path) -> TorchLearningModel:
    """ Load PyTorch learning model from files """
    if not path.exists():
        raise RuntimeError(f"Model folder does not exist: {path}")

    # Load topology
    topology_file = path / "topology.json"
    topology = MlpTopology.from_json(topology_file)
    model = TorchMultiLayerPerceptron(topology)

    # Load weights
    weight_file = path / "weights.pt"
    model.base_model.load_state_dict(torch.load(weight_file, weights_only=True))

    # Load scalers and model
    x_scaler_file, y_scaler_file = scaler_files(path)
    if x_scaler_file.exists() and y_scaler_file.exists():
        x_scaler = joblib.load(x_scaler_file)
        y_scaler = joblib.load(y_scaler_file)
        model.set_scalers(x_scaler, y_scaler)

    return model


def set_seed(seed: int) -> None:
    """ Fix random seed for reproducible results """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
