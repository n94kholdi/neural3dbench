from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from models.base import BaseNetwork
from models.configs import BaseModelConfig
from models.types import ModelCapabilities, ModelInput, ModelOutput


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _pair_of_positive_ints(value: Sequence[int], name: str) -> tuple[int, int]:
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two positive integers.") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain two positive integers.")
    return tuple(_positive_int(item, name) for item in values)


def _nested_tuple(tensor: torch.Tensor):
    if tensor.ndim == 1:
        return tuple(float(value) for value in tensor.tolist())
    return tuple(_nested_tuple(item) for item in tensor)


def _finite_tensor(value, name: str, ndim: int) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array.") from exc
    if tensor.ndim != ndim or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    return tensor


@dataclass
class DeepNURBSConfig(BaseModelConfig):
    """Configuration for a 2D Deep NURBS admissible-field model."""

    degree: tuple[int, int] = (2, 2)
    knot_vectors: tuple[tuple[float, ...], tuple[float, ...]] = (
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    )
    control_points: tuple = field(
        default_factory=lambda: (
            ((0.0, 0.0), (0.0, 0.5), (0.0, 1.0)),
            ((0.5, 0.0), (0.5, 0.5), (0.5, 1.0)),
            ((1.0, 0.0), (1.0, 0.5), (1.0, 1.0)),
        )
    )
    weights: tuple = field(
        default_factory=lambda: ((1.0, 1.0, 1.0),) * 3
    )
    admissible_coefficients: tuple = field(
        default_factory=lambda: ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
    )
    lifting_coefficients: tuple = field(
        default_factory=lambda: (((0.0,),) * 3,) * 3
    )
    output_dim: int = 1
    hidden_dim: int = 50
    num_hidden_layers: int = 1
    activation: str = "cubic_relu"
    representation: str = "PARAMETRIC_SURFACE"

    @classmethod
    def from_parametric_surface(
        cls, data: Mapping[str, Any], **network_overrides: Any
    ) -> "DeepNURBSConfig":
        """Build model geometry from a canonical PARAMETRIC_SURFACE mapping."""
        representation = data.get("representation")
        if representation is not None and str(representation) != "PARAMETRIC_SURFACE":
            raise ValueError("Expected a PARAMETRIC_SURFACE payload.")
        required = (
            "degree",
            "knot_vector_u",
            "knot_vector_v",
            "control_points",
            "weights",
            "admissible_coefficients",
            "lifting_coefficients",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                "PARAMETRIC_SURFACE payload is missing: " + ", ".join(missing)
            )
        lifting = _finite_tensor(data["lifting_coefficients"], "lifting_coefficients", 3)
        payload = {
            "degree": tuple(int(value) for value in data["degree"]),
            "knot_vectors": (data["knot_vector_u"], data["knot_vector_v"]),
            "control_points": data["control_points"],
            "weights": data["weights"],
            "admissible_coefficients": data["admissible_coefficients"],
            "lifting_coefficients": lifting,
            "output_dim": int(lifting.shape[-1]),
        }
        payload.update(network_overrides)
        return cls(**payload)

    def __post_init__(self) -> None:
        self.name = "deep_nurbs"
        self.model_type = "deep_nurbs"
        self.representation = "PARAMETRIC_SURFACE"
        self.degree = _pair_of_positive_ints(self.degree, "degree")
        self.output_dim = _positive_int(self.output_dim, "output_dim")
        self.hidden_dim = _positive_int(self.hidden_dim, "hidden_dim")
        self.num_hidden_layers = _positive_int(
            self.num_hidden_layers, "num_hidden_layers"
        )
        self.activation = self.activation.lower()
        if self.activation not in {"cubic_relu", "tanh"}:
            raise ValueError("activation must be 'cubic_relu' or 'tanh'.")

        try:
            raw_knots = tuple(self.knot_vectors)
        except TypeError as exc:
            raise ValueError("knot_vectors must contain two knot vectors.") from exc
        if len(raw_knots) != 2:
            raise ValueError("knot_vectors must contain two knot vectors.")
        knot_tensors = tuple(
            _finite_tensor(vector, f"knot_vectors[{axis}]", 1)
            for axis, vector in enumerate(raw_knots)
        )
        self.knot_vectors = tuple(_nested_tuple(vector) for vector in knot_tensors)

        control_points = _finite_tensor(self.control_points, "control_points", 3)
        weights = _finite_tensor(self.weights, "weights", 2)
        admissible = _finite_tensor(
            self.admissible_coefficients, "admissible_coefficients", 2
        )
        lifting = _finite_tensor(self.lifting_coefficients, "lifting_coefficients", 3)
        control_shape = tuple(control_points.shape[:2])
        if control_points.shape[-1] != 2:
            raise ValueError("control_points must have shape [Nu, Nv, 2].")
        if tuple(weights.shape) != control_shape:
            raise ValueError(f"weights must have shape {control_shape}.")
        if tuple(admissible.shape) != control_shape:
            raise ValueError(
                f"admissible_coefficients must have shape {control_shape}."
            )
        if tuple(lifting.shape) != (*control_shape, self.output_dim):
            raise ValueError(
                "lifting_coefficients must have shape "
                f"{(*control_shape, self.output_dim)}."
            )
        if torch.any(weights <= 0.0):
            raise ValueError("NURBS weights must be strictly positive.")

        for axis, (degree, knots, control_count) in enumerate(
            zip(self.degree, knot_tensors, control_shape)
        ):
            if degree >= control_count:
                raise ValueError(f"degree[{axis}] must be smaller than its control count.")
            if knots.numel() != control_count + degree + 1:
                raise ValueError(
                    f"knot_vectors[{axis}] must have length "
                    f"control_count + degree + 1 ({control_count + degree + 1})."
                )
            if torch.any(knots[1:] < knots[:-1]):
                raise ValueError(f"knot_vectors[{axis}] must be nondecreasing.")
            if torch.any(torch.unique_consecutive(knots, return_counts=True)[1] > degree + 1):
                raise ValueError(
                    f"knot_vectors[{axis}] multiplicity must not exceed degree + 1."
                )
            if knots[degree].item() != 0.0 or knots[-degree - 1].item() != 1.0:
                raise ValueError(
                    f"knot_vectors[{axis}] must have active domain [0, 1]."
                )
            if torch.any(knots[: degree + 1] != 0.0) or torch.any(
                knots[-degree - 1 :] != 1.0
            ):
                raise ValueError(
                    f"knot_vectors[{axis}] must be clamped at both domain endpoints."
                )

        boundary_coefficients = torch.cat(
            (admissible[0], admissible[-1], admissible[1:-1, 0], admissible[1:-1, -1])
        )
        if not torch.allclose(
            boundary_coefficients,
            torch.zeros_like(boundary_coefficients),
            atol=1e-8,
            rtol=0.0,
        ):
            raise ValueError(
                "admissible_coefficients must be zero on all four control-net boundaries."
            )

        self.control_points = _nested_tuple(control_points)
        self.weights = _nested_tuple(weights)
        self.admissible_coefficients = _nested_tuple(admissible)
        self.lifting_coefficients = _nested_tuple(lifting)


def _bspline_basis(
    values: torch.Tensor, knots: torch.Tensor, degree: int, control_count: int
) -> torch.Tensor:
    basis = (
        (values.unsqueeze(-1) >= knots[:-1])
        & (values.unsqueeze(-1) < knots[1:])
    ).to(values.dtype)
    tiny = torch.finfo(values.dtype).tiny
    for order in range(1, degree + 1):
        left_denominator = knots[order:-1] - knots[: -order - 1]
        right_denominator = knots[order + 1 :] - knots[1:-order]
        left_scale = (values.unsqueeze(-1) - knots[: -order - 1]) / left_denominator.clamp_min(tiny)
        right_scale = (knots[order + 1 :] - values.unsqueeze(-1)) / right_denominator.clamp_min(tiny)
        left_scale = torch.where(left_denominator > 0, left_scale, 0.0)
        right_scale = torch.where(right_denominator > 0, right_scale, 0.0)
        basis = left_scale * basis[..., :-1] + right_scale * basis[..., 1:]

    basis = basis[..., :control_count]
    at_right_endpoint = values == knots[-degree - 1]
    if torch.any(at_right_endpoint):
        endpoint_basis = torch.zeros_like(basis)
        endpoint_basis[..., -1] = 1.0
        basis = torch.where(at_right_endpoint.unsqueeze(-1), endpoint_basis, basis)
    return basis


class DeepNURBS(BaseNetwork):
    """NURBS-admissible neural approximation on a 2D parametric surface."""

    config_class = DeepNURBSConfig

    def __init__(self, config: DeepNURBSConfig | None = None) -> None:
        super().__init__(config=config or DeepNURBSConfig())
        control_points = torch.tensor(self.config.control_points, dtype=torch.float32)
        weights = torch.tensor(self.config.weights, dtype=torch.float32)
        admissible = torch.tensor(
            self.config.admissible_coefficients, dtype=torch.float32
        )
        lifting = torch.tensor(self.config.lifting_coefficients, dtype=torch.float32)
        knots_u = torch.tensor(self.config.knot_vectors[0], dtype=torch.float32)
        knots_v = torch.tensor(self.config.knot_vectors[1], dtype=torch.float32)
        self.register_buffer("control_points", control_points)
        self.register_buffer("nurbs_weights", weights)
        self.register_buffer("admissible_coefficients", admissible)
        self.register_buffer("lifting_coefficients", lifting)
        self.register_buffer("knots_u", knots_u)
        self.register_buffer("knots_v", knots_v)

        self.hidden_layers = nn.ModuleList()
        input_dim = 2
        for _ in range(self.config.num_hidden_layers):
            self.hidden_layers.append(nn.Linear(input_dim, self.config.hidden_dim))
            input_dim = self.config.hidden_dim
        self.output_layer = nn.Linear(input_dim, self.config.output_dim)

    @property
    def required_inputs(self) -> set[str]:
        return {"coordinates"}

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representation="PARAMETRIC_SURFACE",
            requires=set(self.required_inputs),
            supports=set(),
            variable_geometry=False,
            variable_node_count=True,
        )

    def validate_inputs(self, inputs: ModelInput) -> bool:
        super().validate_inputs(inputs)
        coordinates = self._ensure_tensor(inputs.coordinates, name="coordinates")
        if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
            raise ValueError(
                "DeepNURBS coordinates must have shape [B, N, 2], "
                f"got {tuple(coordinates.shape)}."
            )
        if coordinates.shape[0] == 0 or coordinates.shape[1] == 0:
            raise ValueError("coordinates must contain at least one batch and one point.")
        if not coordinates.is_floating_point():
            raise TypeError("coordinates must use a floating-point dtype.")
        if not torch.isfinite(coordinates).all():
            raise ValueError("coordinates must be finite.")
        if torch.any((coordinates < 0.0) | (coordinates > 1.0)):
            raise ValueError("parametric coordinates must lie in [0, 1].")
        return True

    def _evaluate_nurbs(
        self, parametric_coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basis_u = _bspline_basis(
            parametric_coordinates[..., 0],
            self.knots_u,
            self.config.degree[0],
            self.control_points.shape[0],
        )
        basis_v = _bspline_basis(
            parametric_coordinates[..., 1],
            self.knots_v,
            self.config.degree[1],
            self.control_points.shape[1],
        )
        tensor_basis = basis_u.unsqueeze(-1) * basis_v.unsqueeze(-2)
        weighted_basis = tensor_basis * self.nurbs_weights
        rational_basis = weighted_basis / weighted_basis.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(torch.finfo(parametric_coordinates.dtype).tiny)
        physical_coordinates = torch.einsum(
            "bnuv,uvc->bnc", rational_basis, self.control_points
        )
        admissible_field = torch.einsum(
            "bnuv,uv->bn", rational_basis, self.admissible_coefficients
        ).unsqueeze(-1)
        lifting_field = torch.einsum(
            "bnuv,uvc->bnc", rational_basis, self.lifting_coefficients
        )
        return physical_coordinates, admissible_field, lifting_field

    def forward(self, inputs: ModelInput) -> ModelOutput:
        self.validate_inputs(inputs)
        parameter = next(self.parameters())
        parametric_coordinates = torch.as_tensor(
            inputs.coordinates, device=parameter.device, dtype=parameter.dtype
        )
        physical_coordinates, admissible_field, lifting_field = self._evaluate_nurbs(
            parametric_coordinates
        )
        neural_field = physical_coordinates
        for layer in self.hidden_layers:
            neural_field = layer(neural_field)
            if self.config.activation == "cubic_relu":
                neural_field = torch.relu(neural_field).pow(3)
            else:
                neural_field = torch.tanh(neural_field)
        neural_field = self.output_layer(neural_field)
        predictions = admissible_field * neural_field + lifting_field
        return ModelOutput(
            predictions=predictions,
            latent=neural_field,
            metadata={
                "parametric_coordinates": parametric_coordinates,
                "physical_coordinates": physical_coordinates,
                "admissible_field": admissible_field,
                "lifting_field": lifting_field,
            },
        )
