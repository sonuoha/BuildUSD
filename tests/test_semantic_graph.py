from types import SimpleNamespace

from buildusd.semantic_graph import (
    DECOMPOSITION_LOGICAL_ONLY,
    DECOMPOSITION_SEMANTIC_MESH_PARTS,
    build_semantic_graph,
)
from buildusd.semantic_enrichment import build_targeted_enrichment_plan


def _cache(*records):
    return SimpleNamespace(instances={record.step_id: record for record in records})


def test_composite_window_without_part_geometry_requests_targeted_enrichment():
    record = SimpleNamespace(
        step_id=42,
        product_id=42,
        guid="2abc",
        name="Window_Main",
        ifc_class="IfcWindow",
        usd_path="/World/Building/Window_Main",
        prototype_path="/World/__Prototypes/WindowType",
        hierarchy=(("Level 01", 12),),
        materials=[SimpleNamespace(Name="Grey")],
        material_ids=[0],
        style_face_groups={},
        attributes={"psets": {"Pset_WindowCommon": {"IsExternal": True}}},
        semantic_parts={},
        detail_mesh=None,
    )

    graph = build_semantic_graph(_cache(record), base_name="Model")

    element = next(node for node in graph["nodes"] if node["id"] == "element:guid:2abc")
    assert element["properties"]["decomposition_quality"] == DECOMPOSITION_LOGICAL_ONLY
    assert graph["enrichment_requests"] == [
        {
            "kind": "targeted_decomposition",
            "reason": "Composite element has inferred logical parts but no part-level geometry.",
            "guid": "2abc",
            "step_id": 42,
            "ifc_class": "IFCWINDOW",
            "current_usd_prim": "/World/Building/Window_Main",
            "requested_parts": ["Frame", "Glazing", "Hardware"],
            "suggested_detail_objects": ["2abc", 42],
        }
    ]

    part_labels = {
        node["label"] for node in graph["nodes"] if node["type"] == "LogicalPart"
    }
    assert part_labels == {"Frame", "Glazing", "Hardware"}


def test_semantic_parts_are_geometry_backed_without_enrichment_request():
    record = SimpleNamespace(
        step_id=7,
        product_id=7,
        guid="win-semantic",
        name="Window_Split",
        ifc_class="IfcWindow",
        usd_path="/World/Building/Window_Split",
        prototype_path=None,
        hierarchy=(),
        materials=[],
        material_ids=[],
        style_face_groups={},
        attributes={},
        semantic_parts={"Frame": {}, "Glazing": {}},
        semantic_part_prim_paths={
            "Frame": "/World/Building/Window_Split/Geom/Frame",
            "Glazing": "/World/Building/Window_Split/Geom/Glazing",
        },
        detail_mesh=None,
    )

    graph = build_semantic_graph(_cache(record), base_name="Model")

    element = next(
        node for node in graph["nodes"] if node["id"] == "element:guid:win-semantic"
    )
    assert (
        element["properties"]["decomposition_quality"]
        == DECOMPOSITION_SEMANTIC_MESH_PARTS
    )
    assert graph["enrichment_requests"] == []
    glazing = next(node for node in graph["nodes"] if node["label"] == "Glazing")
    assert (
        glazing["properties"]["represented_by"]
        == "/World/Building/Window_Split/Geom/Glazing"
    )


def test_targeted_enrichment_plan_prefers_guid_and_builds_conversion_options():
    graph = {
        "enrichment_requests": [
            {
                "guid": "2abc",
                "step_id": 42,
                "ifc_class": "IFCWINDOW",
                "current_usd_prim": "/World/Window",
                "requested_parts": ["Frame", "Glazing"],
            },
            {
                "guid": None,
                "step_id": 77,
                "ifc_class": "IFCDOOR",
                "current_usd_prim": "/World/Door",
                "requested_parts": ["Panel", "Hardware"],
            },
        ]
    }

    plan = build_targeted_enrichment_plan(graph, detail_engine="semantic")

    assert plan.detail_objects == ("2abc", 77)
    options = plan.to_conversion_options()
    assert options.detail_mode is True
    assert options.detail_scope == "object"
    assert options.detail_objects == ("2abc", 77)
    assert options.detail_engine == "semantic"


def test_targeted_enrichment_plan_can_filter_requested_objects():
    graph = {
        "enrichment_requests": [
            {"guid": "keep-me", "step_id": 1, "requested_parts": ["Glazing"]},
            {"guid": "skip-me", "step_id": 2, "requested_parts": ["Frame"]},
        ]
    }

    plan = build_targeted_enrichment_plan(graph, selected_objects=["keep-me"])

    assert plan.detail_objects == ("keep-me",)
    assert len(plan.targets) == 1
    assert plan.targets[0].requested_parts == ("Glazing",)
