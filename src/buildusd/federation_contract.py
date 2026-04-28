from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal


AnchorMode = Optional[Literal["local", "basepoint"]]  # None means "none/default"


@dataclass(frozen=True)
class FederationDecision:
    anchor_mode: AnchorMode
    model_offset_m: Optional[Tuple[float, float, float]]
    georef_origin: Optional[str]  # "ifc_site" | "pbp" | "shared_site" | None


def decide_offsets(
    *,
    anchor_mode: AnchorMode,
    ifc_site_m: Optional[Tuple[float, float, float]],
    pbp_m: Optional[Tuple[float, float, float]],
    shared_site_m: Optional[Tuple[float, float, float]],
) -> FederationDecision:
    """
    Deterministic 2-mode anchoring contract.

    IfcOpenShell with offset-type='negative':
        stage_pos = authored_pos - model_offset

    Modes:
      1) local     : model_offset = IfcSite placement
      2) basepoint : model_offset = PBP if available, else SharedSite
      3) none      : model_offset = None, georef = None (unless lonlat_override is provided elsewhere)
    """
    mode = anchor_mode or None

    if mode is None:
        return FederationDecision(
            anchor_mode=None, model_offset_m=None, georef_origin=None
        )

    if mode == "local":
        if ifc_site_m is None:
            return FederationDecision(
                anchor_mode="local", model_offset_m=(0.0, 0.0, 0.0), georef_origin=None
            )
        return FederationDecision(
            anchor_mode="local", model_offset_m=ifc_site_m, georef_origin="ifc_site"
        )

    # mode == "basepoint"
    if pbp_m is not None:
        return FederationDecision(
            anchor_mode="basepoint", model_offset_m=pbp_m, georef_origin="pbp"
        )
    if shared_site_m is None:
        return FederationDecision(
            anchor_mode="basepoint",
            model_offset_m=(0.0, 0.0, 0.0),
            georef_origin=None,
        )
    return FederationDecision(
        anchor_mode="basepoint",
        model_offset_m=shared_site_m,
        georef_origin="shared_site",
    )
