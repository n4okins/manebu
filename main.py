import jax
import jax.numpy as jnp
import jraph
import evojax
from evojax.task.base import TaskState, VectorizedTask
from evojax.policy.base import PolicyState, PolicyNetwork

from torchvision import datasets
from flax import linen as nn
from flax.struct import dataclass
import numpy as np


if __name__ == "__main__":
    print(jax.default_backend())