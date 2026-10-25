"""Optimizer base class: the parameter list and the zero_grad/step contract."""

from __future__ import annotations


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError
