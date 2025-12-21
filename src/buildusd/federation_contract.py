from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal


AnchorMode = Optional[Literal["local", "site"]]  # None means "none/default"


@dataclass(frozen=True)
class FederationDecision:
    anchor_mode: AnchorMode
    model_offset_m: Optional[Tuple[float, float, float]]
    georef_origin: Optional[str]  # "pbp" | "shared_site" | None


def decide_offsets(
    *,
    anchor_mode: AnchorMode,
    pbp_m: Optional[Tuple[float, float, float]],
    shared_site_m: Optional[Tuple[float, float, float]],
) -> FederationDecision:
    """
    Deterministic 3-mode anchoring contract.

    IfcOpenShell with offset-type='negative':
        stage_pos = authored_pos - model_offset

    Modes:
      1) local: model_offset = PBP, georef = PBP
      2) site : model_offset = SharedSite, georef = SharedSite
      3) none : model_offset = None, georef = None (unless lonlat_override is provided elsewhere)
    """
    mode = anchor_mode or None

    if mode is None:
        return FederationDecision(
            anchor_mode=None, model_offset_m=None, georef_origin=None
        )

    if mode == "local":
        if pbp_m is None:
            return FederationDecision(
                anchor_mode="local", model_offset_m=None, georef_origin="pbp"
            )
        return FederationDecision(
            anchor_mode="local", model_offset_m=pbp_m, georef_origin="pbp"
        )

    # mode == "site"
    if shared_site_m is None:
        return FederationDecision(
            anchor_mode="site", model_offset_m=None, georef_origin="shared_site"
        )
    return FederationDecision(
        anchor_mode="site", model_offset_m=shared_site_m, georef_origin="shared_site"
    )
