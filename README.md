IFC → USD Converter (Federated)

Overview
- Converts IFC files to USD with prototypes, materials, and instances.
- Optionally adds WGS84 geolocation attributes to /World (when anchoring or lon/lat overrides are supplied and geospatial mode is enabled).
- Provides a separate federation CLI (`python -m buildusd.federate`) that assembles per-file stages into project master files without touching conversion outputs.
- Authors IFC properties/quantities as USD attributes under a BIMData namespace.

Requirements
- Python ≥ 3.11
- Shared dependencies (see `pyproject.toml`):
  - `ifcopenshell==0.8.3.post2`
  - `pyproj==3.7.2` (for CRS transforms)
  - `numpy`, `click`, `rich`, etc.
- **Kit mode (default)** – no standalone `usd-core` wheel required. Install Omniverse Kit (``pip install --extra-index-url https://pypi.nvidia.com omniverse-kit``) so `omni.client` and Kit's pxr are available.
- **Offline mode (`--offline`)** – install a standalone USD build (e.g. ``pip install usd-core``). All paths must be local; `omniverse://` URIs are rejected and checkpointing is skipped.

Support matrix (tested)
- OS: Windows 10/11, Ubuntu 22.04 (headless OK).
- Python: 3.11, 3.12.
- IfcOpenShell: 0.8.3.post2.
- USD bindings: Omniverse Kit pxr (Kit 105/106), usd-core 24.08.
- pythonocc: optional; OCC detail requires an OCC-enabled ifcopenshell build.

Environment
- Windows requires the Microsoft Visual C++ 2015–2022 Redistributable x64.
- Ensure your virtual environment is active before running.

Quick start (offline)
- Download a small public IFC (e.g., Duplex_A_20110907.ifc from common samples) into `data/input/`.
- Convert locally (no Kit):
  `python -m buildusd --offline --input data/input/Duplex_A_20110907.ifc`
- Expected: stages and layers under `data/output/Duplex_A_20110907/`; no Nucleus access.
- Additional IFC samples are available at https://github.com/youshengCode/IfcSampleFiles (see that repository’s license; attribute and comply with its terms when using those files).

Quick start (Nucleus / Kit)
- Accept Kit EULA, install Kit, and ensure an `omniverse://` endpoint is reachable.
- Convert and checkpoint:
  `python -m buildusd --input omniverse://server/Projects/IFC/Duplex_A.ifc --checkpoint`
- Expected: authored layers on Nucleus; headless Kit session auto-starts.

Install
- Create/activate venv and install dependencies per your workflow (e.g., ``pip install -e .`` or ``uv sync``).
- **Kit mode**
  - Accept the Kit EULA once (PowerShell ``set OMNI_KIT_ACCEPT_EULA=yes``, bash ``export OMNI_KIT_ACCEPT_EULA=yes``).
  - Install Kit: ``pip install --extra-index-url https://pypi.nvidia.com omniverse-kit``.
  - Optional: ``python -c "from omni.kit_app import KitApp; KitApp().shutdown(); print('Omniverse ready')"`` to verify the runtime.
  - The converter auto-starts a headless Kit session whenever an `omniverse://` path is encountered.
- **Offline mode**
  - Install ``usd-core`` (or another pxr build) alongside ifcopenshell.
- Run ``python -m buildusd ...`` from the repo root, or ``pip install -e .`` for a global CLI.
- Legacy invocations like ``python -m ifc_converter`` continue to work via a compatibility shim.
- INFO logs show which directory or Nucleus path is scanned and each IFC file as it starts processing (`PYTHONUNBUFFERED=1` for unbuffered output).

Mode selection & environment variables
- The converter can author USD via two bindings:
  - **Kit mode** (default). Used whenever any supplied path is `omniverse://` _or_ when `BUILDUSD_DEFAULT_USD_MODE` (legacy `IFC_CONVERTER_DEFAULT_USD_MODE`) resolves to `kit`. Requires Omniverse Kit (`omni.client`), automatically boots Kit in headless mode, and enables Nucleus features such as checkpoints.
  - **Offline mode**. Activated by providing `--offline` on the CLI, `offline=True` in the Python API, or setting `BUILDUSD_DEFAULT_USD_MODE=offline` (legacy `IFC_CONVERTER_DEFAULT_USD_MODE=offline`). All paths must be local filesystem locations. Nucleus checkpoint requests are ignored and `omniverse://` inputs raise a `ValueError`.
- Mode-precedence rules:
  1. Explicit CLI/programmatic `offline=True` wins.
  2. Otherwise, if any input/output/manifest path starts with `omniverse://`, Kit mode is chosen.
  3. Otherwise, the environment variable `BUILDUSD_DEFAULT_USD_MODE` (`kit` by default, falls back to `IFC_CONVERTER_DEFAULT_USD_MODE`) decides the binding.
- Exclusion handling honours the mode: `--exclude` takes bare stems or names with `.ifc` and skips them case-insensitively during directory scans (local paths or Nucleus directories).
- Relevant environment variables:
  - `BUILDUSD_DEFAULT_USD_MODE` / `IFC_CONVERTER_DEFAULT_USD_MODE` - `kit` (default) or `offline`; establishes the initial USD binding when the process starts.
  - `OMNI_KIT_ACCEPT_EULA` – set to `yes` to suppress Kit's interactive EULA prompt during headless launches.
  - `PYTHONUNBUFFERED` – optional; keep at `1` to stream logs without buffering during long conversions.
  - `USD_FORCE_MODULE_NAME` – honoured by pxr when present; useful if your USD distribution installs under a different module alias.

Usage (CLI)
- Single IFC file:
  - python -m buildusd --input C:\\path\\to\\file.ifc
- Directory, specific names:
  - python -m buildusd --input C:\\path\\to\\dir --ifc-names A.ifc B.ifc
- Directory, all files:
  - python -m buildusd --input C:\\path\\to\\dir --all
- Offline conversion (local-only, no Kit):
  - python -m buildusd --offline --input C:\\path\\to\\dir --all
- Directory, all files excluding drafts:
  - python -m buildusd --input C:\\path\\to\\dir --all --exclude DraftModel TempIFC
- Checkpoint authored layers on Nucleus:
  - python -m buildusd --input omniverse://server/Projects/IFC --all --checkpoint
- Custom CRS (default EPSG:7855):
  - python -m buildusd --input C:\\path\\to\\dir --all --map-coordinate-system EPSG:XXXX
- Manifest-driven base points / federated routing:
  - python -m buildusd --input C:\\path\\to\\dir --all --manifest src/buildusd/config/sample_manifest.json
  - python -m buildusd --input C:\\path\\to\\dir --all --manifest src/buildusd/config/sample_manifest.yaml
- Assemble masters after conversion:
  - python -m buildusd.federate --stage-root data/output --manifest src/buildusd/config/sample_manifest.json
  - python -m buildusd.federate --stage-root data/output --manifest src/buildusd/config/sample_manifest.yaml --masters-root data/federated --rebuild
  - python -m buildusd --input C:\\path\\to\\dir --all --manifest src/buildusd/config/sample_manifest.yaml --federate
- Federation CLI options (`python -m buildusd.federate`):
  - `--stage-root`, `--stage`, `--masters-root`, `--manifest`, `--parent-prim`, `--map-coordinate-system`, `--anchor-mode`, `--frame`, `--offline`, `--rebuild`
- Nucleus (omniverse://) paths work for files or directories:
  - python -m buildusd --input omniverse://server/Projects/IFC --all
- Detail routing examples:
  - python -m buildusd --detail-mode --detail-engine default   # IFC subcomponents first, OCC fallback for all products
  - python -m buildusd --detail-mode --detail-engine occ       # OCC only for all products (skip subcomponents)
  - python -m buildusd --detail-mode --detail-engine semantic  # IFC subcomponents only (no OCC fallback) for all products
  - python -m buildusd --detail-mode --detail-scope object --detail-objects 1265 ubd7n32hksiop  # detail only specific STEP ids / GUIDs
  - python -m buildusd --detail-mode --detail-engine occ --detail-scope object --detail-objects 1265  # OCC-only detail for targeted objects
  - PowerShell note: GUIDs with `$` must be quoted or escaped, e.g. `--detail-objects '0jNViHeUb9$QjXG30GXDQy'` or ``--detail-objects 0jNViHeUb9`$QjXG30GXDQy``.
- CLI detail flags (defaults/behavior):
  - `--detail-mode`: off by default; enables the detail pipeline.
  - `--detail-scope`: optional; defaults to `all` when omitted. Use `object` with `--detail-objects`.
  - `--detail-objects`: space-separated STEP ids and/or GUIDs; implies `--detail-mode` and `--detail-scope object` when supplied.
  - `--detail-engine`: `default` (semantic first, OCC fallback), `occ|opencascade` (OCC only), `semantic|ifc-subcomponents|ifc-parts` (semantic only). If OCC is unavailable, the engine falls back to semantic with a warning.
  - Shell quoting: in PowerShell, `$` expands variables; wrap GUIDs in single quotes or escape `$` with a backtick.
- Update meters-per-unit metadata on an existing USD stage/layer (no IFC conversion):
  - python -m buildusd --set-stage-unit "omniverse://server/Projects/file.usdc" --stage-unit-value 0.001
- Update up-axis metadata on an existing USD stage/layer (no IFC conversion):
  - python -m buildusd --set-stage-up-axis "omniverse://server/Projects/file.usdc" --stage-up-axis Z
- 2D annotation extraction (off by default):
  - python -m buildusd --input C:\\path\\to\\dir --include-2d
- Annotation curve widths (use with --include-2d):
  - python -m buildusd --input C:\\path\\to\\dir --include-2d --annotation-width-default 15mm
  - python -m buildusd --input C:\\path\\to\\dir --include-2d --annotation-width-rule width=0.02,layer=Survey*,curve=*Centerline*
  - python -m buildusd --input C:\\path\\to\\dir --include-2d --annotation-width-config src/buildusd/config/sample_annotation_widths.json
  - python -m buildusd --input C:\\path\\to\\dir --include-2d --annotation-width-config src/buildusd/config/sample_annotation_widths.yaml

CLI call signatures
```bash
python -m buildusd [--input PATH] [--output PATH] [options]
python -m buildusd.federate --stage-root PATH --manifest MANIFEST [options]
```

CLI options (summary)
- Input/output: `--input`, `--output`, `--ifc-names`, `--exclude`, `--all`, `--manifest`
- Execution: `--offline`, `--checkpoint`, `--usd-format`, `--usd-auto-binary-threshold-mb`, `--map-coordinate-system`, `--geospatial-mode`
- 2D: `--include-2d`, `--annotation-width-default`, `--annotation-width-rule`, `--annotation-width-config`
- Anchoring/federation: `--anchor-mode`, `--federate`, `--frame` (federation only)
- Detail: `--detail-mode`, `--detail-scope`, `--detail-objects`, `--detail-engine`, `--enable-semantic-subcomponents`, `--semantic-tokens`
- Utilities: `--set-stage-unit`, `--stage-unit-value`, `--set-stage-up-axis`, `--stage-up-axis`

CLI reference (full)
```text
python -m buildusd
  --map-coordinate-system, --map-epsg   EPSG code or CRS string for map eastings/northings
  --input PATH                          IFC file or directory (default: repo root)
  --output PATH                         Output directory for USD artifacts
  --manifest PATH                       Manifest (YAML/JSON) for base points and masters
  --ifc-names NAMES...                  Specific IFC files to process in a directory
  --exclude NAMES...                    IFC file names to skip
  --all                                 Process all .ifc files in the input directory
  --checkpoint                          Create Nucleus checkpoints (omniverse:// only)
  --offline                             Force standalone USD (no Kit); local paths only
  --set-stage-unit PATH                 Update metersPerUnit on an existing layer/stage
  --stage-unit-value FLOAT              metersPerUnit value for --set-stage-unit
  --set-stage-up-axis PATH              Update upAxis on an existing layer/stage
  --stage-up-axis {X|Y|Z}               upAxis value for --set-stage-up-axis
  --annotation-width-default VALUE      Default annotation width (e.g. 15mm)
  --annotation-width-rule SPEC          Width override rule (repeatable)
  --annotation-width-config PATH        Width config file (repeatable)
  --include-2d                          Enable 2D annotation extraction
  --anchor-mode {local|basepoint|none}  Anchor mode for model offsets
  --federate                            Run federation after conversion
  --frame {projected|geodetic}          Federation frame (used with --federate)
  --geospatial-mode {auto|usd|omni|none} Geospatial metadata mode
  --usd-format {usdc|usda|usd|auto}     Output USD format
  --usd-auto-binary-threshold-mb FLOAT  Re-export as usdc above this size (MB)
  --detail-mode                         Enable detail pipeline
  --detail-scope {all|object}           Scope for detail meshes (default: all)
  --detail-objects STEP_OR_GUID...      Targets for object-scoped detail
  --detail-engine {default|occ|opencascade|semantic|ifc-subcomponents|ifc-parts}
                                       Detail engine routing
  --enable-semantic-subcomponents       Enable semantic subcomponent splitting
  --semantic-tokens PATH                JSON file of semantic tokens
```

```text
python -m buildusd.federate
  --stage-root PATH                     Root directory containing converted stages
  --stage PATHS...                      Specific stage files to federate
  --masters-root PATH                   Output directory for federated masters
  --manifest PATH                       Manifest describing federation targets
  --parent-prim PATH                    Parent prim for payloads (default: /World)
  --map-coordinate-system EPSG          Fallback CRS when manifest omits projected_crs
  --anchor-mode {local|basepoint|none}  Match anchoring used by converted stages
  --frame {projected|geodetic}          Federation frame for delta computation
  --offline                             Standalone USD mode (no Kit)
  --rebuild                             Rebuild masters from scratch
```

Model offsets & anchoring
- Stages stay in meters (metersPerUnit=1.0).
- Iterator tessellation runs with use-world-coords=False; placements come from the iterator transform.
- Per file we resolve a model offset when --anchor-mode is set:
  - local -> IfcSite.ObjectPlacement (meters)
  - basepoint -> Project Base Point (PBP) if available, else Survey Point (SP), else (0,0,0) with a warning
  - none -> no model-offset (geospatial metadata only if a lon/lat override is supplied)
- Offsets are baked into geometry via ifcopenshell model-offset; no USD XformOps are authored for anchoring.
- The GeometrySettingsManager applies offset-type (default negative) to the raw offset and pushes the signed value into all ifcopenshell settings objects.
- MapConversion grid rotation (when applicable) is applied via ifcopenshell model-rotation (quaternion), not USD XformOps.
- Each IFC file gets its own resolved offset; offsets are not shared across files.

Usage (VS Code)
- Press F5 and pick one of the provided launch configurations in .vscode/launch.json.
- Modify args there to suit your inputs.

Usage (Python)
- from buildusd import convert
- results = convert("path/to/file.ifc", output_dir="data/output")  # returns List[ConversionResult]
- convert("omniverse://server/Projects/file.ifc", output_dir="omniverse://server/USD/output")
- convert("path/to/file.ifc", output_dir="data/output", checkpoint=True)  # omniverse:// required for checkpoints
- from buildusd import ConversionOptions
- convert("path/to/file.ifc", output_dir="data/output", options=ConversionOptions(include_2d=True))
- from buildusd.api import set_stage_unit
- set_stage_unit("omniverse://server/Projects/file.usdc", meters_per_unit=0.001)
- from buildusd.api import set_stage_up_axis
- set_stage_up_axis("omniverse://server/Projects/file.usdc", axis="Z")

ConversionOptions examples (programmatic)
- Detail all with OCC fallback after subcomponents:
  - `options = ConversionOptions(detail_mode=True, detail_scope="all", detail_engine="default")`
- Detail specific objects (mixed ids/guids) via OCC only:
  - `options = ConversionOptions(detail_mode=True, detail_scope="object", detail_objects=(1265, "UBD7N32HKSiop"), detail_engine="occ")`
- Semantic-only detail (no OCC fallback):
  - `options = ConversionOptions(detail_mode=True, detail_scope="all", detail_engine="semantic")`
- Geometry overrides (safe subset only):
  - `options = ConversionOptions(detail_mode=True, detail_scope="all", geom_overrides={"mesher-linear-deflection": 0.5, "mesher-angular-deflection": 5})`
- Include 2D annotation extraction (default is off):
  - `options = ConversionOptions(include_2d=True)`

Manifest schema
- Sample manifests live in `src/buildusd/config/sample_manifest.{json,yaml}`. Keep the same structure (masters, base points, CRS). Add a jsonschema alongside your manifests if you want automated validation (e.g., `manifest.schema.json`) and validate with `buildusd validate-manifest` when available.

Outputs
- Per-IFC stages and layers are written to data/output:
  - <name>.usda (stage)
  - prototypes/<name>_prototypes.usda
  - materials/<name>_materials.usda
  - instances/<name>_instances.usda
  - geometry2d/<name>_geometry2d.usda (when present; captured 2D alignment/annotation curves; requires --include-2d or include_2d=True)
    - /World/<file>_Instances preserves the IFC spatial hierarchy (Project/Site/Storey/Class).
    - Optional grouping variants (see src/buildusd/process_usd.py:author_instance_grouping_variant) can reorganize instances on demand without losing the canonical hierarchy.
  - caches/<name>.json stores serialized instance metadata for later regrouping sessions.
- Optional federated masters (run `python -m buildusd.federate --manifest ...` after conversion):
  - Creates master stage(s) defined in the manifest without overwriting per-file outputs.
  - Each converted stage is referenced beneath `/World/<safe_name>` so you can compose projects on demand.

Materials
- IFC render precedence respected: geometry `IfcStyledItem` styles first, then material presentation (`IfcMaterialDefinitionRepresentation`), then shape aspects, then type. When no MDR is present, we also try `ifcopenshell.util.representation.get_material_style` for the associated material.
- `IfcSurfaceStyleRendering` maps to PreviewSurface: baseColor from SurfaceColour (else DiffuseColour), opacity from Transparency, roughness from SpecularRoughness/specular level, metallic from ReflectanceMethod, emissive from EmissiveColour/SelfLuminous. SurfaceStyleWithTextures/ImageTexture set the baseColor texture when present; UVs from IfcIndexedPolygonalTextureMap are authored as `primvars:st`.
- Names drop literal “Undefined” and add a closest CSS color hint when available (`webcolors` preferred; small fallback palette otherwise).
- Multiple materials → face subsets; iterator materials are not force-overridden beyond IFC precedence. Single-material meshes bind the resolved style when only one material id exists and no face-level subsets are defined.
- Texture safety: only http/https and relative file:// paths are used; absolute file:// paths are ignored for safety.

License
- GPL-3.0. This is a copyleft license; consuming projects must comply with GPL terms. If you need a different license for your use case, discuss with the maintainers.


Geometry overrides (advanced)
- You can supply a limited set of geometry overrides via `ConversionOptions.geom_overrides` or `ConversionSettings.geom_overrides`.
- Supported keys align with ifcopenshell/occ settings (e.g., `mesher-linear-deflection`, `mesher-angular-deflection`, `compute-normals`).
- Core pipeline settings are fixed internally and ignored if provided here: `use-world-coords`, `model-offset`, `offset-type`, `use-python-opencascade`.
- Anchoring/model offsets remain controlled by the converter; overrides are merged on top of the defaults where safe.
Detail / remesh
- `enable_high_detail_remesh` defaults to False; the iterator mesh is the base geometry. `--detail-mode` runs the detail pipeline without remeshing unless explicitly enabled.
- Detail scope is optional; it defaults to `all` when omitted. Use `object` with `--detail-objects`.
- OCC detail meshes author under `/World/__PrototypesDetail` (and instance overrides when scoped); the base iterator tessellation remains the primary geometry path.
- Detail engine routing:
  - `--detail-engine default` (default) tries IFC subcomponents first, then falls back to OCC.
  - `--detail-engine occ|opencascade` skips subcomponents and goes straight to OCC.
  - `--detail-engine semantic|ifc-subcomponents|ifc-parts` runs IFC subcomponent splitting only (no OCC fallback).
  - `--detail-objects` accepts mixed STEP ids and GUIDs when `--detail-scope object` is used (e.g. `--detail-objects 1265 ubd7n32hksiop`). Supplying `--detail-objects` auto-enables `--detail-mode` and forces `--detail-scope object`.
  - PowerShell quoting: use single quotes or escape `$` in GUIDs (e.g. `'0jNViHeUb9$QjXG30GXDQy'` or ``0jNViHeUb9`$QjXG30GXDQy``).
  - Env caps: `OCC_DETAIL_FACE_CAP` skips OCC detail when face count exceeds the cap; `OCC_CANONICAL_MAP_FACE_CAP` skips canonical map building when faces exceed the cap or the mesh is single-material with no item ids.

Annotation Curve Width Overrides
- 2D curve widths are only evaluated when 2D extraction is enabled (`--include-2d` or API option).
- Control the `UsdGeom.BasisCurves` widths authored in geometry2d layers via `--annotation-width-default`, repeated `--annotation-width-rule`, or config files supplied with `--annotation-width-config`.
- Widths accept numeric values in stage units (`0.015`) or include a unit suffix (`12mm`, `1.5cm`, `0.01m`). A separate `unit` key is also accepted in configuration mappings.
- Rule filters support `layer` (matches the IFC stem used for the geometry2d layer), `curve` (annotation name), `hierarchy` (any label or `/`-joined path in the spatial hierarchy), and `step_id`. Glob-style (`fnmatch`) patterns are applied case-insensitively.
- Rules are evaluated in order: configuration files are loaded first, then the CLI default, followed by any CLI rule expressions. Later matches override earlier ones.
- Example JSON configuration (see `src/buildusd/config/sample_annotation_widths.json`):

```json
{
  "default": "0.015",
  "layers": {
    "Survey*": "12mm"
  },
  "curves": {
    "*Control*": "0.02"
  },
  "layer_curves": {
    "Alignment*": {
      "Centerline*": {"width": 18, "unit": "mm"},
      "Offset*": "0.01"
    }
  },
  "hierarchies": {
    "*Level 01*": "0.012"
  }
}
```

- Example YAML configuration (see `src/buildusd/config/sample_annotation_widths.yaml`):

```yaml
default: 0.015

layers:
  "Survey*": 12mm
  "Alignment*":
    width: 18
    unit: mm

curves:
  "*Centerline*": 0.02

layer_curves:
  "Alignment*":
    "Offset*": 0.01

hierarchies:
  "*Level 01*": 0.012
```

Units and Geospatial
- Per-file stages author WGS84 reference on /World and /World/Geospatial (OmniGeospatial referencePosition) when anchoring or lon/lat overrides are supplied and geospatial mode is enabled. Projected anchors are stored as `ifc:anchorProjected` customData when available.
- Federated masters created via `buildusd.federate` are authored with metersPerUnit=1.0 (meters). Payloads are not rescaled; a log line indicates alignment or mismatch.

Geo Anchoring
- Conversion and federation expose `--anchor-mode` to control model offsets and anchor metadata. `local` uses IfcSite placement, `basepoint` uses PBP/SP, and `none` skips model offsets (geospatial metadata only if a lon/lat override is supplied).
- Geodetic metadata (lon/lat/height) is derived from the chosen anchor (using `pyproj` when available) and written on /World and /World/Geospatial alongside the `ifc:` attributes; metersPerUnit remains 1.0.
- Example invocations: `python -m buildusd --anchor-mode basepoint ...` for conversion, `python -m buildusd --anchor-mode none ...` to skip model offsets, and `python -m buildusd.federate --anchor-mode basepoint ...` to keep federated masters aligned in the same frame.

IFC Metadata as USD Attributes
- IFC psets/qtos are authored as attributes (not customData) using:
  - BIMData:Psets:<PsetName>:<PropName>
  - BIMData:QTO:<QtoName>:<PropName>
- Types are inferred (Bool/Int/Double and arrays; fallback String).

Federated Stage Behavior (via `buildusd.federate`)
- Each converted USD stage is referenced as a payload under `/World/<safe_name>` in the manifest-selected master stage.
- The payload targets the stage's default prim so additional `/World` nesting is avoided when possible.
- The federation output is idempotent: re-running adds missing payloads; use `--rebuild` to recreate the master stage from scratch.
- If `defaults.overall_master_name` is set, a top-level overall master is built by payloading the per-site master stages.
- Running `python -m buildusd --federate` after conversion uses the same routing logic as `buildusd.federate` and respects `--anchor-mode`/`--frame` for alignment.

Programmatic Use
- `from buildusd import api` exposes structured helpers. `api.ConversionSettings` and `api.convert()` mirror the CLI; `api.FederationSettings` and `api.federate_stages()` do the same for master assembly; `api.apply_stage_anchor_transform()` anchors custom USD stages consistently.
- `api.CONVERSION_DEFAULTS` / `api.FEDERATION_DEFAULTS` expose the packaged defaults, and `api.DEFAULT_CONVERSION_OPTIONS` offers a ready-to-clone baseline for geometry harvesting.
- Anchor modes accept `"local"`, `"basepoint"`, or `None`/`"none"`; `none` skips model offsets and only stamps geospatial metadata when a lon/lat override is supplied.
- main(argv=None) and parse_args(argv=None) accept a list of tokens to drive from scripts/notebooks.
- Use `ConversionSettings(include_2d=True)` or `ConversionOptions(include_2d=True)` to opt into 2D annotation extraction.

```python
from buildusd import api

settings = api.ConversionSettings(
    input_path="path/to/file.ifc",
    output_dir="data/output",
    include_2d=True,
    manifest_path="src/buildusd/config/sample_manifest.yaml",
)
results = api.convert(settings)

federation_settings = api.FederationSettings(
    stage_paths=[r.stage_path for r in results if r.stage_path],
    masters_root="data/federated",
    manifest_path="src/buildusd/config/sample_manifest.yaml",
    anchor_mode="basepoint",
    frame="projected",
)
api.federate_stages(federation_settings)
```
Manifest Schema
- defaults: Global fallback for master name, projected/geodetic CRS, base point, shared site base point, and optional `file_revision` used for checkpoint notes/tags.
  - `overall_master_name` (or `overall_master`) enables an overall master stage.
  - `overall_base_point` / `overall_shared_site_base_point` set the overall federation origin.
- masters: Named per-site federated stages with optional CRS/base point overrides and `file_revision`. The master `base_point` is used as the site federation origin; falls back to shared site base point if missing.
- files: Match rules (name or glob pattern) that choose a master, override CRS/base point/lonlat, and provide a per-file `file_revision`.

Notes
- JSON manifests work immediately; YAML manifests require installing PyYAML.
- Sample manifest templates live at:
  - src/buildusd/config/sample_manifest.json (JSON with `_comment` helper fields)
  - src/buildusd/config/sample_manifest.yaml (YAML with inline comments)
  Copy one of them locally (e.g. to src/buildusd/config/manifest.yaml) when preparing project-specific settings. The real manifest remains untracked by design and can be loaded from local paths or omniverse:// URIs.
- 2D annotation contexts (e.g. alignment strings in IfcAnnotation) are preserved only when 2D extraction is enabled. If the ifcopenshell geometry iterator rejects an annotation context, the pipeline emits a warning and falls back to manual curve extraction so the data still lands in the 2D geometry layer.


Troubleshooting
- pxr ImportError with _tf/_usd DLLs on Windows: install latest VC++ redistributable x64.
- CRS conversions require pyproj; if missing, WGS84 attributes won’t be authored.

Examples
- NVIDIA CAD Converter export of the same tunnel segment shows gaps and lost detail when tessellating the IFC input.

![NVIDIA CAD converter output showing geometry loss](data/input/img/CAD_converter.png)

- Our IFC pipeline preserves full segment detail and materials while authoring clean instance hierarchies.

![Pipeline output preserving object integrity](data/input/img/Pipeline.png)
