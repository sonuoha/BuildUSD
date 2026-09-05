"""Render the BuildUSD functional architecture as a clean SVG and vector PDF.

The validated SysML v2 model remains the semantic source of truth.  This script
creates a presentation view inspired by function-flow block diagrams: compact
functions, explicit ports, named state flows, and local auxiliary-information
buses with no crossing primary flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, Sequence

from reportlab.pdfgen import canvas


WIDTH = 1800
HEIGHT = 1120

BG = "#FFFFFF"
SURFACE = "#F8FAFC"
HEADER = "#E5E7EB"
BORDER = "#9CA3AF"
TEXT = "#1F2937"
MUTED = "#64748B"
FLOW = "#9B2C3B"
AUX = "#6B7280"
CONVERSION = "#2563EB"
FEDERATION = "#15803D"
ENRICHMENT = "#7E22CE"


@dataclass(frozen=True)
class FunctionBlock:
    title: str
    module: str
    input_name: str
    output_name: str


@dataclass(frozen=True)
class Lane:
    code: str
    name: str
    accent: str
    input_state: str
    output_state: str
    auxiliary: str
    blocks: tuple[FunctionBlock, ...]
    states: tuple[str, ...]
    auxiliary_targets: tuple[int, ...]


LANES = (
    Lane(
        code="F1",
        name="Convert IFC assets",
        accent=CONVERSION,
        input_state="IFC source set + conversion request",
        output_state="USD stages + semantic graph + result",
        auxiliary="manifest + options + cancellation + runtime mode",
        blocks=(
            FunctionBlock(
                "Interpret intent",
                "api.py | cli.py | conversion.py",
                "request",
                "intent",
            ),
            FunctionBlock(
                "Discover sources",
                "conversion.py | io_utils.py",
                "source set",
                "selected IFC",
            ),
            FunctionBlock(
                "Load IFC models",
                "io_utils.py | ifcopenshell",
                "selected IFC",
                "parsed model",
            ),
            FunctionBlock(
                "Resolve georeference",
                "georef_resolution.py | resolve_frame.py",
                "parsed model",
                "resolved model",
            ),
            FunctionBlock(
                "Build prototypes",
                "process_ifc.py | detail processors",
                "resolved model",
                "prototype cache",
            ),
            FunctionBlock(
                "Author USD layers", "process_usd.py", "prototype cache", "layer set"
            ),
            FunctionBlock(
                "Anchor and publish",
                "process_usd.py | semantic_graph.py",
                "layer set",
                "published assets",
            ),
        ),
        states=(
            "interpreted intent",
            "selected IFC set",
            "parsed IFC model",
            "resolved model",
            "prototype cache",
            "USD layer set",
        ),
        auxiliary_targets=(0, 1, 3, 5, 6),
    ),
    Lane(
        code="F2",
        name="Federate USD stages",
        accent=FEDERATION,
        input_state="converted USD stage set + federation request",
        output_state="validated federated project stage",
        auxiliary="manifest + frame + anchor policy + existing-target hints",
        blocks=(
            FunctionBlock(
                "Plan federation",
                "federation_orchestrator.py",
                "stages",
                "routed stages",
            ),
            FunctionBlock(
                "Inspect anchors",
                "federation_builder.py",
                "routed stages",
                "anchor evidence",
            ),
            FunctionBlock(
                "Align payloads",
                "federation_builder.py | geodetic_federation.py",
                "anchor evidence",
                "aligned payloads",
            ),
            FunctionBlock(
                "Compose site masters",
                "federation_builder.py",
                "aligned payloads",
                "site masters",
            ),
            FunctionBlock(
                "Compose project master",
                "federation_orchestrator.py",
                "site masters",
                "project master",
            ),
            FunctionBlock(
                "Validate federation",
                "federation_builder.py",
                "project master",
                "validated project",
            ),
        ),
        states=(
            "routed stage set",
            "inspected payload set",
            "aligned payload set",
            "site master set",
            "federated project",
        ),
        auxiliary_targets=(0, 2, 4, 5),
    ),
    Lane(
        code="F3",
        name="Run targeted enrichment job",
        accent=ENRICHMENT,
        input_state="pending job + semantic graph",
        output_state="detail artifacts + job result + archive",
        auxiliary="semantic graph + detail options + cache policy",
        blocks=(
            FunctionBlock("Claim job", "worker.py", "pending job", "claimed job"),
            FunctionBlock(
                "Resolve targets",
                "semantic_enrichment.py | jobs.py",
                "claimed job",
                "detail plan",
            ),
            FunctionBlock(
                "Check detail cache", "jobs.py", "detail plan", "cache decision"
            ),
            FunctionBlock(
                "Convert targeted detail",
                "jobs.py -> conversion pipeline",
                "cache miss",
                "detail artifacts",
            ),
            FunctionBlock(
                "Publish manifest",
                "jobs.py",
                "detail artifacts",
                "replacement manifest",
            ),
            FunctionBlock("Archive outcome", "worker.py", "manifest", "job result"),
        ),
        states=(
            "claimed job",
            "targeted detail plan",
            "cache miss",
            "detail artifact set",
            "replacement manifest",
        ),
        auxiliary_targets=(1, 2, 3),
    ),
)


class SvgPainter:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = []

    def rounded_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        radius: float,
        fill: str,
        stroke: str,
        stroke_width: float = 1.0,
    ) -> None:
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1.0,
    ) -> None:
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        width: float = 1.0,
        dashed: bool = False,
    ) -> None:
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        self.items.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"{dash}/>'
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        color: str,
        width: float = 1.0,
        dashed: bool = False,
    ) -> None:
        dash = ' stroke-dasharray="7 6"' if dashed else ""
        encoded = " ".join(f"{x},{y}" for x, y in points)
        self.items.append(
            f'<polyline points="{encoded}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round"{dash}/>'
        )

    def polygon(
        self,
        points: Sequence[tuple[float, float]],
        fill: str,
        stroke: str,
        stroke_width: float = 1.0,
    ) -> None:
        encoded = " ".join(f"{x},{y}" for x, y in points)
        self.items.append(
            f'<polygon points="{encoded}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def circle(self, x: float, y: float, radius: float, fill: str) -> None:
        self.items.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}"/>')

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float,
        color: str = TEXT,
        bold: bool = False,
        anchor: str = "start",
    ) -> None:
        weight = "500" if bold else "400"
        self.items.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'
        )

    def finish(self) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc">\n'
            '<title id="title">BuildUSD functional architecture</title>\n'
            '<desc id="desc">Three aligned function-flow lanes for IFC conversion, USD federation, and targeted enrichment.</desc>\n'
            + "\n".join(self.items)
            + "\n</svg>\n"
        )


class PdfPainter:
    def __init__(
        self,
        pdf: canvas.Canvas,
        page_width: float,
        page_height: float,
        model_width: float,
        model_height: float,
    ) -> None:
        self.pdf = pdf
        self.page_width = page_width
        self.page_height = page_height
        self.scale = min(
            (page_width - 48) / model_width, (page_height - 48) / model_height
        )
        self.offset_x = (page_width - model_width * self.scale) / 2
        self.offset_y = (page_height - model_height * self.scale) / 2
        self.model_height = model_height

    def _x(self, x: float) -> float:
        return self.offset_x + x * self.scale

    def _y(self, y: float) -> float:
        return self.offset_y + (self.model_height - y) * self.scale

    def rounded_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        radius: float,
        fill: str,
        stroke: str,
        stroke_width: float = 1.0,
    ) -> None:
        fill_enabled = fill != "none"
        self.pdf.setDash()
        if fill_enabled:
            self.pdf.setFillColor(fill)
        self.pdf.setStrokeColor(stroke)
        self.pdf.setLineWidth(stroke_width * self.scale)
        self.pdf.roundRect(
            self._x(x),
            self._y(y + h),
            w * self.scale,
            h * self.scale,
            radius * self.scale,
            fill=1 if fill_enabled else 0,
            stroke=1,
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str = "none",
        stroke_width: float = 1.0,
    ) -> None:
        self.pdf.setDash()
        self.pdf.setFillColor(fill)
        self.pdf.setStrokeColor(fill if stroke == "none" else stroke)
        self.pdf.setLineWidth(stroke_width * self.scale)
        self.pdf.rect(
            self._x(x),
            self._y(y + h),
            w * self.scale,
            h * self.scale,
            fill=1,
            stroke=0 if stroke == "none" else 1,
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str,
        width: float = 1.0,
        dashed: bool = False,
    ) -> None:
        self.pdf.setStrokeColor(color)
        self.pdf.setLineWidth(width * self.scale)
        self.pdf.setDash(
            7 * self.scale, 6 * self.scale
        ) if dashed else self.pdf.setDash()
        self.pdf.line(self._x(x1), self._y(y1), self._x(x2), self._y(y2))

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        color: str,
        width: float = 1.0,
        dashed: bool = False,
    ) -> None:
        self.pdf.setStrokeColor(color)
        self.pdf.setLineWidth(width * self.scale)
        self.pdf.setDash(
            7 * self.scale, 6 * self.scale
        ) if dashed else self.pdf.setDash()
        path = self.pdf.beginPath()
        x0, y0 = points[0]
        path.moveTo(self._x(x0), self._y(y0))
        for x, y in points[1:]:
            path.lineTo(self._x(x), self._y(y))
        self.pdf.drawPath(path, fill=0, stroke=1)

    def polygon(
        self,
        points: Sequence[tuple[float, float]],
        fill: str,
        stroke: str,
        stroke_width: float = 1.0,
    ) -> None:
        self.pdf.setDash()
        self.pdf.setFillColor(fill)
        self.pdf.setStrokeColor(stroke)
        self.pdf.setLineWidth(stroke_width * self.scale)
        path = self.pdf.beginPath()
        x0, y0 = points[0]
        path.moveTo(self._x(x0), self._y(y0))
        for x, y in points[1:]:
            path.lineTo(self._x(x), self._y(y))
        path.close()
        self.pdf.drawPath(path, fill=1, stroke=1)

    def circle(self, x: float, y: float, radius: float, fill: str) -> None:
        self.pdf.setDash()
        self.pdf.setFillColor(fill)
        self.pdf.setStrokeColor(fill)
        self.pdf.circle(self._x(x), self._y(y), radius * self.scale, fill=1, stroke=0)

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float,
        color: str = TEXT,
        bold: bool = False,
        anchor: str = "start",
    ) -> None:
        font = "Helvetica-Bold" if bold else "Helvetica"
        font_size = size * self.scale
        self.pdf.setFont(font, font_size)
        self.pdf.setFillColor(color)
        width = self.pdf.stringWidth(value, font, font_size)
        draw_x = self._x(x)
        if anchor == "middle":
            draw_x -= width / 2
        elif anchor == "end":
            draw_x -= width
        self.pdf.drawString(draw_x, self._y(y), value)


def draw_port(
    painter: SvgPainter | PdfPainter, x: float, y: float, direction: str
) -> None:
    if direction == "in":
        points = ((x - 7, y - 7), (x + 2, y), (x - 7, y + 7))
    else:
        points = ((x - 2, y - 7), (x + 7, y), (x - 2, y + 7))
    painter.polygon(points, BG, BORDER, 1.3)


def draw_terminal(
    painter: SvgPainter | PdfPainter, x: float, y: float, direction: str, color: str
) -> None:
    if direction == "in":
        points = (
            (x - 12, y - 8),
            (x + 2, y - 8),
            (x + 10, y),
            (x + 2, y + 8),
            (x - 12, y + 8),
        )
    else:
        points = (
            (x - 10, y),
            (x - 2, y - 8),
            (x + 12, y - 8),
            (x + 12, y + 8),
            (x - 2, y + 8),
        )
    painter.polygon(points, color, color, 1.0)


def draw_function_block(
    painter: SvgPainter | PdfPainter,
    x: float,
    y: float,
    block: FunctionBlock,
    w: float = 190,
    h: float = 115,
) -> None:
    painter.rounded_rect(x + 4, y + 5, w, h, 6, "#E2E8F0", "#E2E8F0", 0)
    painter.rounded_rect(x, y, w, h, 6, SURFACE, BORDER, 1.4)
    painter.rect(x, y, w, 35, HEADER, BORDER, 1.0)
    painter.text(x + 12, y + 23, block.title, 13, TEXT, True)
    port_y = y + 70
    draw_port(painter, x, port_y, "in")
    draw_port(painter, x + w, port_y, "out")
    painter.text(x + 12, port_y + 5, block.input_name, 10, MUTED)
    painter.text(x + w - 12, port_y + 5, block.output_name, 10, MUTED, False, "end")
    painter.text(x + 12, y + 101, block.module, 8.5, MUTED)


def evenly_spaced_positions(count: int) -> list[float]:
    left = 135.0
    right = 1475.0
    if count == 1:
        return [(left + right) / 2]
    step = (right - left) / (count - 1)
    return [left + index * step for index in range(count)]


def draw_lane(painter: SvgPainter | PdfPainter, lane: Lane, top: float) -> None:
    lane_x = 55
    lane_w = 1690
    lane_h = 260
    painter.rounded_rect(lane_x, top, lane_w, lane_h, 10, BG, "#CBD5E1", 1.2)
    painter.rect(lane_x, top, 8, lane_h, lane.accent)
    painter.text(lane_x + 22, top + 28, f"{lane.code}  {lane.name}", 15, TEXT, True)

    bus_y = top + 58
    painter.line(145, bus_y, 1688, bus_y, AUX, 1.2, True)
    draw_terminal(painter, 145, bus_y, "in", AUX)
    painter.text(165, bus_y - 9, lane.auxiliary, 9.5, MUTED)

    block_y = top + 102
    positions = evenly_spaced_positions(len(lane.blocks))
    flow_y = block_y + 70

    draw_terminal(painter, 86, flow_y, "in", lane.accent)
    painter.line(96, flow_y, positions[0] - 8, flow_y, FLOW, 2.0)
    painter.text(86, top + 89, f"INPUT  {lane.input_state}", 9.5, TEXT, True)

    for index, (x, block) in enumerate(zip(positions, lane.blocks)):
        if index in lane.auxiliary_targets:
            port_x = x + 95
            painter.line(port_x, bus_y, port_x, block_y - 1, AUX, 1.0, True)
            painter.polygon(
                (
                    (port_x - 6, block_y - 7),
                    (port_x + 6, block_y - 7),
                    (port_x, block_y + 1),
                ),
                BG,
                BORDER,
                1.0,
            )
        draw_function_block(painter, x, block_y, block)
        if index < len(lane.blocks) - 1:
            next_x = positions[index + 1]
            painter.line(x + 197, flow_y, next_x - 8, flow_y, FLOW, 2.0)
            midpoint = (x + 197 + next_x - 8) / 2
            painter.circle(midpoint, flow_y, 3.0, FLOW)

    final_x = positions[-1] + 197
    painter.line(final_x, flow_y, 1698, flow_y, FLOW, 2.0)
    draw_terminal(painter, 1710, flow_y, "out", lane.accent)
    painter.text(1710, top + 89, f"OUTPUT  {lane.output_state}", 9.5, TEXT, True, "end")

    if lane.code == "F3":
        cache_x = positions[2] + 95
        publish_x = positions[4] + 95
        bypass_y = block_y + 135
        painter.polyline(
            (
                (cache_x, block_y + 115),
                (cache_x, bypass_y),
                (publish_x, bypass_y),
                (publish_x, block_y + 115),
            ),
            AUX,
            1.2,
            True,
        )
        painter.text(
            (cache_x + publish_x) / 2,
            bypass_y - 8,
            "cache hit - reuse detail artifacts",
            9,
            MUTED,
            False,
            "middle",
        )


def draw_architecture(painter: SvgPainter | PdfPainter) -> None:
    painter.rect(0, 0, WIDTH, HEIGHT, BG)
    painter.text(55, 58, "BuildUSD functional architecture", 27, TEXT, True)
    painter.text(
        55,
        88,
        "SysML v2 action and information-flow view derived from the validated textual model",
        12,
        MUTED,
    )

    legend_x = 1220
    legend_y = 42
    painter.rounded_rect(legend_x, legend_y, 525, 63, 7, SURFACE, "#CBD5E1", 1.0)
    painter.rounded_rect(legend_x + 16, legend_y + 15, 70, 30, 4, HEADER, BORDER, 1.0)
    painter.text(legend_x + 51, legend_y + 35, "function", 9, TEXT, True, "middle")
    painter.line(
        legend_x + 105, legend_y + 30, legend_x + 180, legend_y + 30, FLOW, 2.0
    )
    painter.text(legend_x + 190, legend_y + 34, "primary state flow", 9, MUTED)
    painter.line(
        legend_x + 320, legend_y + 30, legend_x + 390, legend_y + 30, AUX, 1.2, True
    )
    painter.text(legend_x + 400, legend_y + 34, "auxiliary information", 9, MUTED)

    for lane, top in zip(LANES, (125, 420, 715)):
        draw_lane(painter, lane, top)

    painter.text(55, 1028, "System boundary", 11, MUTED, True)
    painter.rounded_rect(45, 108, 1710, 900, 12, "none", "#64748B", 1.6)
    painter.text(
        55,
        1065,
        "Port labels name the information state at each function boundary. Module paths show current implementation allocation.",
        10,
        MUTED,
    )
    painter.text(1745, 1085, "BuildUSD v0.2.0", 9, MUTED, False, "end")


def render_svg(path: Path) -> None:
    painter = SvgPainter(WIDTH, HEIGHT)
    draw_architecture(painter)
    path.write_text(painter.finish(), encoding="utf-8")


def render_pdf(path: Path) -> None:
    page_width = 18 * 72
    page_height = 11 * 72
    pdf = canvas.Canvas(str(path), pagesize=(page_width, page_height))
    pdf.setTitle("BuildUSD functional architecture")
    pdf.setAuthor("BuildUSD project")
    painter = PdfPainter(pdf, page_width, page_height, WIDTH, HEIGHT)
    draw_architecture(painter)
    pdf.showPage()
    pdf.save()


def main() -> None:
    architecture_dir = Path(__file__).resolve().parent
    repository_root = architecture_dir.parent.parent
    render_svg(architecture_dir / "buildusd-functional-architecture.svg")
    output_dir = repository_root / "output" / "pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    render_pdf(output_dir / "buildusd-functional-architecture-neat.pdf")


if __name__ == "__main__":
    main()
