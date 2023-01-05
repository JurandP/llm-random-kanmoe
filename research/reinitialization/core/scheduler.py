from abc import ABC, abstractmethod

from research.reinitialization.core.pruner import BasePruner
from attr import define


class BaseScheduler(ABC):
    @abstractmethod
    def step(self):
        ...


@define
class DelayedConstScheduler(BaseScheduler):
    pruner: BasePruner
    n_steps_prune: int
    prob: float
    delay: int = 0
    current_step = 0

    def step(self):
        if (
            self.current_step % self.n_steps_prune == 0
            and self.current_step >= self.delay
        ):
            self.pruner.prune(self.prob)
        self.current_step += 1


@define
class MagnitudeStatScheduler(BaseScheduler):
    pruner: BasePruner
    n_steps_prune: int
    prob: float
    n_steps_log_recycle_hist: int
    n_steps_log_magnitude: int
    n_steps_hist_all: int
    delay: int = 0
    current_step = 0
    init: str = "kaiming"

    def step(self):
        self.pruner.log_recently_pruned_magnitude(self.current_step)

        if (
            self.current_step % self.n_steps_prune == 0
            and self.current_step >= self.delay
        ):
            self.pruner.prune(self.prob, self.init)

        if (
            self.init == "grad"
            and self.current_step >= self.delay
            and self.current_step % self.n_steps_prune == 1
        ):
            print("!!! GRAD CORRECT will be!!!")
            self.pruner.grad_correct()

        if self.current_step % self.n_steps_log_recycle_hist == 0:
            self.pruner.log_recycle_magnitude(self.current_step)

        if self.current_step % self.n_steps_log_magnitude == 0:
            self.pruner.log_magnitude(self.current_step)

        if self.current_step % self.n_steps_hist_all == 0:
            self.pruner.log_hist_all_weights(self.current_step)
        self.current_step += 1
