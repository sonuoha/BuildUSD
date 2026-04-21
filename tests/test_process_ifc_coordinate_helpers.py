from types import SimpleNamespace

from buildusd import process_ifc


class FakeOperation:
    def __init__(self, kind: str, **attrs):
        self._kind = kind
        for key, value in attrs.items():
            setattr(self, key, value)

    def is_a(self, kind):
        if isinstance(kind, tuple):
            return self._kind in kind
        return self._kind == kind


class FakeContext:
    def __init__(self, *, context_type, dimension, operations):
        self.ContextType = context_type
        self.CoordinateSpaceDimension = dimension
        self.HasCoordinateOperation = operations


class FakeIfcFile:
    def __init__(self, contexts):
        self._contexts = contexts

    def by_type(self, type_name):
        if type_name == "IfcGeometricRepresentationContext":
            return self._contexts
        return []


def test_extract_map_conversion_prefers_model_3d_operation():
    fallback = FakeContext(
        context_type="Plan",
        dimension=2,
        operations=[
            FakeOperation(
                "IfcRigidOperation",
                FirstCoordinate=1.0,
                SecondCoordinate=2.0,
                Height=3.0,
            )
        ],
    )
    model = FakeContext(
        context_type="Model",
        dimension=3,
        operations=[
            FakeOperation(
                "IfcMapConversion",
                Eastings=100.0,
                Northings=200.0,
                OrthogonalHeight=30.0,
                XAxisAbscissa=0.0,
                XAxisOrdinate=1.0,
                Scale=2.0,
            )
        ],
    )

    result = process_ifc.extract_map_conversion(FakeIfcFile([fallback, model]))

    assert result is not None
    assert result.eastings == 100.0
    assert result.northings == 200.0
    assert result.orthogonal_height == 30.0
    assert result.x_axis_abscissa == 0.0
    assert result.x_axis_ordinate == 1.0
    assert result.scale == 2.0


def test_extract_map_conversion_falls_back_to_first_supported_operation():
    fallback = FakeContext(
        context_type="Plan",
        dimension=2,
        operations=[
            FakeOperation(
                "IfcRigidOperation",
                FirstCoordinate=10.0,
                SecondCoordinate=20.0,
                Height=3.5,
            )
        ],
    )

    result = process_ifc.extract_map_conversion(FakeIfcFile([fallback]))

    assert result is not None
    assert result.eastings == 10.0
    assert result.northings == 20.0
    assert result.orthogonal_height == 3.5
    assert result.x_axis_abscissa == 1.0
    assert result.x_axis_ordinate == 0.0
    assert result.scale == 1.0


def test_resolve_absolute_matrix_preserves_iterator_transform():
    matrix = tuple(float(i) for i in range(16))
    shape = SimpleNamespace(transformation=SimpleNamespace(matrix=matrix))

    result = process_ifc.resolve_absolute_matrix(
        shape,
        element=None,
        model_offset=(1.0, 2.0, 3.0),
        model_offset_type="negative",
    )

    assert result == matrix
