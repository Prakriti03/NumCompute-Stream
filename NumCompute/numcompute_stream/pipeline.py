"""
pipeline.py —  pipeline for chaining transformers and estimators.

Provides a Transformer and Estimator base class, and a Pipeline class
that chains multiple processing steps in order. Compatible with any
object that follows the fit/transform or fit/predict interface.

Notes
-----
Time complexity depends on the steps used. Pipeline itself adds O(s)
overhead for s steps, excluding the cost of each step.
Space complexity depends on intermediate transformed arrays.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


class Transformer:
    """Base class for all transformers.
    Subclasses must override transform(). fit() and fit_transform()
    are provided but can also be overridden.
    """

    def fit(self, X: NDArray) -> "Transformer":
        """Learn parameters from X. Override in subclass if needed.
        Parameters
        ----------
        X : array, shape (n, d) — training data
        Returns
        -------
        self
        """
        return self

    def transform(self, X: NDArray) -> NDArray:
        """Apply transformation to X. Must be overridden in subclass.
        Parameters
        ----------
        X : array, shape (n, d)
        Returns
        -------
        array — transformed data
        Raises
        ------
        NotImplementedError — if not overridden
        """
        raise NotImplementedError("Transformer must implement transform().")

    def fit_transform(self, X: NDArray) -> NDArray:
        """Fit to X then transform it.
        Parameters
        ----------
        X : array, shape (n, d)
        Returns
        -------
        array — transformed data
        """
        return self.fit(X).transform(X)


class Estimator:
    """Base class for all estimators.
    Subclasses must override predict(). fit() is provided but
    can also be overridden.
    """
    def fit(self, X: NDArray, y: NDArray) -> "Estimator":
        """Fit to training data. Override in subclass.
        Parameters
        ----------
        X : array, shape (n, d) — features
        y : array, shape (n,) — targets
        Returns
        -------
        self
        """
        return self

    def predict(self, X: NDArray) -> NDArray:
        """Predict on new data. Must be overridden in subclass.
        Parameters
        ----------
        X : array, shape (n, d)
        Returns
        -------
        array, shape (n,) — predictions
        Raises
        ------
        NotImplementedError — if not overridden
        """
        raise NotImplementedError("Estimator must implement predict().")

class Pipeline:
    """Chain transformers with an optional final estimator.

    Each step is a (name, object) tuple. All steps except the last
    must have fit() and transform(). The last step must have either
    transform() or predict().

    Parameters
    ----------
    steps : list of (str, object) tuples
        Named processing steps. Names must be unique.

    Raises
    ------
    TypeError  — if steps is not a list, or step format is wrong,
                 or a step is missing required methods
    ValueError — if steps is empty or names are duplicated
    """
    def __init__(self, steps: list[tuple[str, object]]):
        if not isinstance(steps, list):
            raise TypeError("steps must be a list.")
        if len(steps) == 0:
            raise ValueError("steps must not be empty.")
        self.steps = steps
        self._check_steps()

    def _check_steps(self):
        """Validate step format, unique names, and required interfaces."""
        names = []
        for i, step in enumerate(self.steps):
            if not isinstance(step, tuple) or len(step) != 2:
                raise TypeError("Each step must be a (name, object) tuple.")
            name, obj = step
            if not isinstance(name, str) or name.strip() == "":
                raise ValueError("Step name must be a non-empty string.")
            if name in names:
                raise ValueError(f"Duplicate step name: '{name}'.")
            names.append(name)
            if not hasattr(obj, "fit"):
                raise TypeError(f"Step '{name}' must have fit().")
            if i < len(self.steps) - 1 and not hasattr(obj, "transform"):
                raise TypeError(f"Non-final step '{name}' must have transform().")

        last_name, last_obj = self.steps[-1]
        if not hasattr(last_obj, "transform") and not hasattr(last_obj, "predict"):
            raise TypeError(
                f"Final step '{last_name}' must have transform() or predict()."
            )

    def fit(self, X: NDArray, y: NDArray | None = None) -> "Pipeline":
        """Fit all steps sequentially on training data.
        Each intermediate step is fit and transformed before passing
        data to the next step. The final step is fit only.
        Parameters
        ----------
        X : array — input features
        y : array or None — targets, passed to final step if it has predict()
        Returns
        -------
        self
        """
        Xt = np.asarray(X)
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt) if hasattr(step, "fit_transform") else step.fit(Xt).transform(Xt)
        _, final = self.steps[-1]
        if hasattr(final, "predict") and y is not None:
            final.fit(Xt, y)
        else:
            final.fit(Xt)
        return self

    def fit_transform(self, X: NDArray, y: NDArray | None = None) -> NDArray:
        """Fit all steps and return transformed output of the final step.
        Parameters
        ----------
        X : array — input features
        y : array or None — targets
        Returns
        -------
        array — transformed data from final step
        """
        Xt = np.asarray(X)
        for _, step in self.steps[:-1]:
            Xt = step.fit_transform(Xt) if hasattr(step, "fit_transform") else step.fit(Xt).transform(Xt)
        _, final = self.steps[-1]
        if hasattr(final, "fit_transform"):
            return final.fit_transform(Xt)
        return final.fit(Xt).transform(Xt)

    def transform(self, X: NDArray) -> NDArray:
        """Transform data through all steps (used on test data after fit).
        Parameters
        ----------
        X : array — input features
        Returns
        -------
        array — transformed data
        Raises
        ------
        TypeError — if any step does not support transform()
        """
        Xt = np.asarray(X)
        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError(f"Step '{name}' does not support transform().")
            Xt = step.transform(Xt)
        return Xt

    def predict(self, X: NDArray) -> NDArray:
        """Transform through intermediate steps then predict with final estimator.
        Parameters
        ----------
        X : array — input features
        Returns
        -------
        array — predictions from final estimator
        Raises
        ------
        TypeError — if the final step does not support predict()
        """
        Xt = np.asarray(X)
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        final_name, final = self.steps[-1]
        if not hasattr(final, "predict"):
            raise TypeError(f"Final step '{final_name}' does not support predict().")
        return final.predict(Xt)

    def __repr__(self) -> str:
        names = [name for name, _ in self.steps]
        return f"Pipeline(steps={names})"