from abc import ABC, abstractclassmethod

class Module(ABC):
    """Base class for all classes in frame work to ensure the same attributes and common function names."""

    def __init__(self) -> None:
        # Attributes to hold input and outputs
        self.input = None
        self.output = None
    
    @abstractclassmethod
    def forward(self):
        pass

    @abstractclassmethod
    def backward(self):
        pass
