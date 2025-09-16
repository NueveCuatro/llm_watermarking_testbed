from abc import ABC, abstractmethod

class BaseWm(ABC):
    """
    This is an abstract base class for the watermaking module. 

    To create a subclass you to implement :
    - <insert> :        Insterst the wayto insert the mark into a modality (data, model loss...)
    - <extract> :       Extract the mark from the said modality
    """

    def __init__(self, opt, modality):
        """
        Initializes the BaseWm class

        Args: 
        - opt (Option class) : is the option dict
        - modality (Model, Model.loss or Data objects) : is the variable with the modalities to modify.
        """
        super().__init__()
        self.opt = opt
        self.modality = modality

    @abstractmethod
    def insert(self, modality):
        """
        This is the method that will be used to insert the mark into the modality.
        The modality can be either be data, the model istself (the mark in the weights), or the loss.
        - If modality is the data, insert will modify the data at training or testing time before the forward pass
        - If the modality is the model, insert will modify the vanilla model to add layers or change the architecture
        - If the modality is the loss, insert will modify the loss defined in the model.py
        """
        pass

    @abstractmethod
    def extract(self, modality):
        pass
