import sys
import types

# Stub ifcopenshell to avoid heavy dependency during unit tests.
if "ifcopenshell" not in sys.modules:
    ifcopenshell_stub = types.ModuleType("ifcopenshell")
    ifcopenshell_stub.__path__ = []
    ifc_util = types.ModuleType("ifcopenshell.util")
    ifc_util.__path__ = []
    ifc_util.element = types.ModuleType("ifcopenshell.util.element")
    ifc_util.representation = types.ModuleType("ifcopenshell.util.representation")
    ifc_geom = types.ModuleType("ifcopenshell.geom")

    sys.modules["ifcopenshell"] = ifcopenshell_stub
    sys.modules["ifcopenshell.util"] = ifc_util
    sys.modules["ifcopenshell.util.element"] = ifc_util.element
    sys.modules["ifcopenshell.util.representation"] = ifc_util.representation
    sys.modules["ifcopenshell.geom"] = ifc_geom

    ifcopenshell_stub.util = ifc_util
    ifcopenshell_stub.geom = ifc_geom

from buildusd.georef_resolution import (
    CoordinateReference,
    GeoreferencePlan,
    LocalizationPlan,
    MapConversionData,
    PlacementStats,
    build_ifc_georeferencing_plan,
    derive_stage_georeference,
    decide_georeference_plan,
    decide_localization_plan,
    resolve_coordinate_reference,
)


def test_localization_skips_offset_when_candidate_would_push_model_far_from_origin():
    points = ((10.0, 20.0, 0.0), (50.0, -5.0, 0.0))
    before = PlacementStats(sample_count=2, max_abs_m=50.0, space="local")

    plan = decide_localization_plan(
        points_m=points,
        metrics_before=before,
        requested_anchor_mode="basepoint",
        requested_anchor_source="pbp",
        requested_anchor_world_m=(318197.2518, 5815160.4723, 0.0),
        candidate_model_offset_m=(318197.2518, 5815160.4723, 0.0),
    )

    assert not plan.apply_model_offset
    assert plan.skip_reason == "candidate_does_not_improve_localization"


def test_localization_applies_when_candidate_rebases_projected_geometry():
    points = ((338294.0, 5805921.0, 0.0), (338431.5, 5805975.5, 0.0))
    before = PlacementStats(sample_count=2, max_abs_m=5805975.5, space="global")

    plan = decide_localization_plan(
        points_m=points,
        metrics_before=before,
        requested_anchor_mode="local",
        requested_anchor_source="coordinate_operation",
        requested_anchor_world_m=(338294.0, 5805921.0, 0.0),
        candidate_model_offset_m=(338294.0, 5805921.0, 0.0),
    )

    assert plan.apply_model_offset
    assert plan.applied_model_offset_m == (338294.0, 5805921.0, 0.0)
    assert plan.metrics_after is not None
    assert plan.metrics_after.space == "local"


def test_georef_plan_falls_back_to_coordinate_operation_when_requested_offset_is_skipped():
    requested = LocalizationPlan(
        requested_anchor_mode="basepoint",
        requested_anchor_source="pbp",
        requested_anchor_world_m=(318197.2518, 5815160.4723, 0.0),
        candidate_model_offset_m=(318197.2518, 5815160.4723, 0.0),
        apply_model_offset=False,
        applied_model_offset_m=None,
        model_offset_type=None,
        skip_reason="candidate_does_not_improve_localization",
        metrics_before=PlacementStats(sample_count=2, max_abs_m=100.0, space="local"),
        metrics_after=PlacementStats(
            sample_count=2, max_abs_m=5815160.0, space="global"
        ),
    )
    coordinate_operation = LocalizationPlan(
        requested_anchor_mode="local",
        requested_anchor_source="coordinate_operation",
        requested_anchor_world_m=(338294.0, 5805921.0, 0.0),
        candidate_model_offset_m=(0.0, 0.0, 0.0),
        apply_model_offset=False,
        applied_model_offset_m=None,
        model_offset_type=None,
        skip_reason="candidate_zero",
        metrics_before=PlacementStats(sample_count=2, max_abs_m=100.0, space="local"),
        metrics_after=PlacementStats(sample_count=2, max_abs_m=100.0, space="local"),
    )

    plan = decide_georeference_plan(
        localization=requested,
        coordinate_operation_localization=coordinate_operation,
        coordinate_operation_origin_world_m=(338294.0, 5805921.0, 0.0),
    )

    assert isinstance(plan, GeoreferencePlan)
    assert plan.effective_anchor_mode == "local"
    assert plan.georef_origin == "coordinate_operation"
    assert plan.projected_anchor_world_m == (338294.0, 5805921.0, 0.0)


class _IfcEmpty:
    def by_type(self, _name):
        return []


class _IfcEntity:
    def __init__(self, type_name, **attrs):
        self._type_name = type_name
        for key, value in attrs.items():
            setattr(self, key, value)

    def is_a(self, type_name):
        if isinstance(type_name, tuple):
            return self._type_name in type_name
        return self._type_name == type_name


class _IfcByType:
    def __init__(self, **entries):
        self._entries = entries

    def by_type(self, name):
        return self._entries.get(name, [])


def test_coordinate_reference_uses_external_crs_when_ifc_discovery_is_missing():
    ref = resolve_coordinate_reference(
        _IfcEmpty(),
        map_conv=None,
        external_projected_crs="EPSG:7855",
    )

    assert ref.projected_crs == "EPSG:7855"
    assert ref.projected_crs_source == "external"
    assert not ref.has_authoritative_operation


def test_build_plan_resolves_coordinate_operation_crs_and_stage_origin():
    source_crs = _IfcEntity("IfcProjectedCRS", Name="EPSG:4326")
    target_crs = _IfcEntity("IfcProjectedCRS", Name="EPSG:7855")
    operation = _IfcEntity(
        "IfcMapConversion",
        SourceCRS=source_crs,
        TargetCRS=target_crs,
    )
    context = _IfcEntity(
        "IfcGeometricRepresentationContext",
        ContextType="Model",
        CoordinateSpaceDimension=3,
        HasCoordinateOperation=[operation],
    )
    ifc = _IfcByType(IfcGeometricRepresentationContext=[context])
    map_conv = MapConversionData(
        eastings=338294.0,
        northings=5805921.0,
        orthogonal_height=12.5,
        x_axis_abscissa=1.0,
        x_axis_ordinate=0.0,
        scale=1.0,
    )

    plan = build_ifc_georeferencing_plan(
        ifc,
        unit_scale_m=1.0,
        anchor_mode="local",
        default_pbp_world_m=None,
        default_shared_site_world_m=None,
        external_projected_crs="EPSG:28355",
        map_conv=map_conv,
    )

    assert plan.coordinate_reference.source_crs == "EPSG:4326"
    assert plan.coordinate_reference.projected_crs == "EPSG:7855"
    assert plan.coordinate_reference.projected_crs_source == "ifc_coordinate_operation"
    assert plan.coordinate_reference.coordinate_operation_kind == "IfcMapConversion"
    assert plan.coordinate_reference.warnings == (
        "external_projected_crs_conflicts_with_ifc_coordinate_operation",
    )
    assert plan.stage_georef.projected_crs == "EPSG:7855"
    assert plan.stage_georef.stage_origin_projected_m == (
        338294.0,
        5805921.0,
        12.5,
    )
    assert plan.stage_georef.status == "authoritative"
    assert plan.stage_georef.georef_source == "coordinate_operation_stage_origin"


def test_stage_georef_uses_coordinate_operation_origin_when_local_stage_is_not_localized():
    stage_georef = derive_stage_georeference(
        coordinate_reference=CoordinateReference(
            source_crs=None,
            projected_crs="EPSG:7855",
            projected_crs_label="EPSG:7855",
            projected_crs_source="ifc_coordinate_operation",
            coordinate_operation_kind="IfcMapConversion",
            coordinate_operation_source="ifc_coordinate_operation",
            coordinate_operation_origin_world_m=(338294.0, 5805921.0, 0.0),
            has_authoritative_operation=True,
            warnings=(),
        ),
        localization=LocalizationPlan(
            requested_anchor_mode="basepoint",
            requested_anchor_source="pbp",
            requested_anchor_world_m=(318197.2518, 5815160.4723, 0.0),
            candidate_model_offset_m=(318197.2518, 5815160.4723, 0.0),
            apply_model_offset=False,
            applied_model_offset_m=None,
            model_offset_type=None,
            skip_reason="candidate_does_not_improve_localization",
            metrics_before=PlacementStats(
                sample_count=2, max_abs_m=100.0, space="local"
            ),
            metrics_after=PlacementStats(
                sample_count=2, max_abs_m=5815160.0, space="global"
            ),
        ),
        geom_space="local",
        map_conv=MapConversionData(
            eastings=338294.0,
            northings=5805921.0,
            orthogonal_height=0.0,
            x_axis_abscissa=1.0,
            x_axis_ordinate=0.0,
            scale=1.0,
        ),
        unit_scale_m=1.0,
    )

    assert stage_georef.stage_origin_projected_m == (338294.0, 5805921.0, 0.0)
    assert stage_georef.status == "authoritative"
    assert stage_georef.georef_source == "coordinate_operation_stage_origin"


def test_stage_georef_uses_applied_global_offset_without_coordinate_operation():
    stage_georef = derive_stage_georeference(
        coordinate_reference=CoordinateReference(
            source_crs=None,
            projected_crs="EPSG:7855",
            projected_crs_label=None,
            projected_crs_source="external",
            coordinate_operation_kind=None,
            coordinate_operation_source=None,
            coordinate_operation_origin_world_m=None,
            has_authoritative_operation=False,
            warnings=(),
        ),
        localization=LocalizationPlan(
            requested_anchor_mode="basepoint",
            requested_anchor_source="pbp",
            requested_anchor_world_m=(333800.49, 5809101.468, 0.0),
            candidate_model_offset_m=(333800.49, 5809101.468, 0.0),
            apply_model_offset=True,
            applied_model_offset_m=(333800.49, 5809101.468, 0.0),
            model_offset_type="negative",
            skip_reason=None,
            metrics_before=PlacementStats(
                sample_count=2, max_abs_m=5809200.0, space="global"
            ),
            metrics_after=PlacementStats(
                sample_count=2, max_abs_m=100.0, space="local"
            ),
        ),
        geom_space="global",
        map_conv=None,
        unit_scale_m=1.0,
    )

    assert stage_georef.stage_origin_projected_m == (333800.49, 5809101.468, 0.0)
    assert stage_georef.status == "declared"
    assert stage_georef.georef_source == "applied_localization:pbp"
