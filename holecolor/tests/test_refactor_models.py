from holecolor.geometry.models import BufferGeometry, HoleBufferRelation, HoleTierRecord, WaferGeometry


def test_refactor_geometry_models_serialize_to_json() -> None:
    wafer = WaferGeometry(id="wafer-0", center_xy_px=(10.0, 20.0), radius_px=30.0, confidence=0.9, visible_arc_intervals_deg=[(0.0, 180.0)], detection_mode="test")
    buffer = BufferGeometry(id="buffer-0", state="unknown", center_xy_px=None, radius_px=None, confidence=0.0, detection_mode="not_implemented")
    hole = HoleTierRecord(node_id=1, center_xy_px=(5.0, 6.0), radius_px=7.0, confidence=0.8, lattice_i=2, lattice_j=3, tier=1, source_class="anchor")
    relation = HoleBufferRelation(node_id=1, relation="border_unknown")

    assert wafer.to_json()["center_xy_px"] == [10.0, 20.0]
    assert buffer.to_json()["state"] == "unknown"
    assert hole.to_json()["center_xy_px"] == [5.0, 6.0]
    assert relation.to_json()["relation"] == "border_unknown"
