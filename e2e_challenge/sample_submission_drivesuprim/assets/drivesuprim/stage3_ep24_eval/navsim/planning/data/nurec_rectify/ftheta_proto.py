"""Stand-ins for the f-theta protobuf messages the rectifier expects.

The rectification code is taken verbatim from the AlpaSim challenge submission,
which reads its camera model out of ``alpasim_grpc``. That package is not
installed here and pulling it in would drag the whole gRPC stack along for six
fields, so these provide the same surface: attribute access, ``HasField`` and
``CopyFrom``, with protobuf's rule that a submessage counts as present only once
something has been written to it.
"""

from __future__ import annotations

import copy
from typing import List


class LinearCDE:
    """The ``linear_cde`` submessage: a 2x2 affine on the distorted pixel plane."""

    def __init__(self, owner: "FthetaCameraParam" = None,
                 linear_c: float = 1.0, linear_d: float = 0.0, linear_e: float = 0.0):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "linear_c", float(linear_c))
        object.__setattr__(self, "linear_d", float(linear_d))
        object.__setattr__(self, "linear_e", float(linear_e))

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, float(value) if name.startswith("linear_") else value)
        owner = object.__getattribute__(self, "_owner")
        if owner is not None and name.startswith("linear_"):
            # Writing through a submessage marks it present, as protobuf does.
            owner._linear_cde_present = True


class FthetaCameraParam:
    """``sensorsim_pb2.FthetaCameraParam`` as far as the rectifier uses it."""

    def __init__(self):
        self.angle_to_pixeldist_poly: List[float] = []
        self.pixeldist_to_angle_poly: List[float] = []
        self.principal_point_x: float = 0.0
        self.principal_point_y: float = 0.0
        self.max_angle: float = 0.0
        self._linear_cde_present = False
        self._linear_cde = LinearCDE(owner=self)

    @property
    def linear_cde(self) -> LinearCDE:
        return self._linear_cde

    def HasField(self, name: str) -> bool:
        if name != "linear_cde":
            raise ValueError(f"unknown field {name!r}")
        return self._linear_cde_present

    def CopyFrom(self, other: "FthetaCameraParam") -> None:
        self.angle_to_pixeldist_poly = list(other.angle_to_pixeldist_poly)
        self.pixeldist_to_angle_poly = list(other.pixeldist_to_angle_poly)
        self.principal_point_x = float(other.principal_point_x)
        self.principal_point_y = float(other.principal_point_y)
        self.max_angle = float(other.max_angle)
        self._linear_cde_present = other.HasField("linear_cde")
        self._linear_cde = LinearCDE(
            owner=self,
            linear_c=other.linear_cde.linear_c,
            linear_d=other.linear_cde.linear_d,
            linear_e=other.linear_cde.linear_e,
        )
        # CopyFrom must not resurrect an absent submessage.
        self._linear_cde_present = other.HasField("linear_cde")

    def __deepcopy__(self, memo):
        clone = FthetaCameraParam()
        clone.CopyFrom(self)
        return clone


class CameraIntrinsics:
    """The ``intrinsics`` message: a resolution plus one camera-model oneof."""

    def __init__(self, resolution_w: int, resolution_h: int, ftheta_param: FthetaCameraParam):
        self.resolution_w = int(resolution_w)
        self.resolution_h = int(resolution_h)
        self.ftheta_param = ftheta_param

    def WhichOneof(self, name: str) -> str:
        if name != "camera_param":
            raise ValueError(f"unknown oneof {name!r}")
        return "ftheta_param"


class AvailableCamera:
    """The subset of ``AvailableCamerasReturn.AvailableCamera`` that is read."""

    def __init__(self, logical_id: str, intrinsics: CameraIntrinsics):
        self.logical_id = logical_id
        self.intrinsics = intrinsics


def ftheta_param_from_usdz(parameters: dict) -> FthetaCameraParam:
    """Build the camera model from a USDZ ``camera_model.parameters`` block.

    The USDZ carries exactly the fields the rectifier reads, under JSON names:
    ``principal_point`` as a pair, the two polynomials as lists, ``max_angle``
    in radians and ``linear_cde`` as ``[c, d, e]``.
    """
    param = FthetaCameraParam()
    px, py = parameters["principal_point"]
    param.principal_point_x = float(px)
    param.principal_point_y = float(py)
    param.angle_to_pixeldist_poly = [float(v) for v in parameters["angle_to_pixeldist_poly"]]
    param.pixeldist_to_angle_poly = [float(v) for v in parameters["pixeldist_to_angle_poly"]]
    param.max_angle = float(parameters.get("max_angle") or 0.0)
    cde = parameters.get("linear_cde")
    if cde is not None:
        param.linear_cde.linear_c = float(cde[0])
        param.linear_cde.linear_d = float(cde[1])
        param.linear_cde.linear_e = float(cde[2])
    return param


def available_camera_from_usdz(logical_id: str, camera_model: dict) -> AvailableCamera:
    """Wrap a USDZ ``camera_model`` block as the camera message."""
    if camera_model.get("type") != "ftheta":
        raise ValueError(f"{logical_id}: expected an ftheta camera, got {camera_model.get('type')!r}")
    parameters = camera_model["parameters"]
    width, height = parameters["resolution"]
    return AvailableCamera(
        logical_id=logical_id,
        intrinsics=CameraIntrinsics(
            resolution_w=width,
            resolution_h=height,
            ftheta_param=ftheta_param_from_usdz(parameters),
        ),
    )
