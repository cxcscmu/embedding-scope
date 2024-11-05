"""
Specify the trainer interface.
"""

from abc import ABC


class Trainer(ABC):
    """
    Base class for all trainers.
    """

    @abstractmethod
    def train(self):
        """
        Train the model.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        """
        Validate the model.
        """
        raise NotImplementedError

    # @abstractmethod
    # def dispatch(self):
    #     """
    #     Interleave training and validation with checkpoints.
    #     """
    #     raise NotImplementedError
