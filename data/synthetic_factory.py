"""
Synthetic manufacturing data generator.
Builds a realistic knowledge graph with a hidden causal chain
for the RCA demo: bad supplier batch B-442 → defective bearings
→ multiple machines affected.
"""

from datetime import datetime, timedelta

from domain.enums import (
    AlertSeverity,
    EntityType,
    MaintenanceType,
    RelationType,
)
from domain.models import Entity, Relationship


class SyntheticFactory:
    """
    Generates all entities and relationships for the
    manufacturing knowledge graph.

    Usage:
        factory = SyntheticFactory()
        entities, relationships = factory.build()
    """

    def __init__(self) -> None:
        self._entities: list[Entity] = []
        self._relationships: list[Relationship] = []

    def build(self) -> tuple[list[Entity], list[Relationship]]:
        """Generate the full knowledge graph and return (entities, relationships)."""
        self._build_plants()
        self._build_assembly_lines()
        self._build_machines()
        self._build_sensors()
        self._build_personnel()
        self._build_suppliers_and_batches()
        self._build_parts()
        self._build_maintenance_events()
        self._build_alerts_and_defects()
        self._build_materials()
        self._build_process_logs()
        return self._entities, self._relationships

    # ── Helpers ─────────────────────────────────────────

    def _add_entity(self, **kwargs) -> Entity:
        entity = Entity(**kwargs)
        self._entities.append(entity)
        return entity

    def _add_rel(self, source_id: str, target_id: str, relation_type: RelationType, **props) -> Relationship:
        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=props,
        )
        self._relationships.append(rel)
        return rel

    # ── Plants ──────────────────────────────────────────

    def _build_plants(self) -> None:
        self._add_entity(
            id="plant_01",
            name="Plant Floor 1",
            entity_type=EntityType.PLANT,
            properties={
                "location": "Building A, Hyderabad",
                "area_sqft": 50000,
                "shifts": 3,
            },
        )
        self._add_entity(
            id="plant_02",
            name="Plant Floor 2",
            entity_type=EntityType.PLANT,
            properties={
                "location": "Building B, Hyderabad",
                "area_sqft": 35000,
                "shifts": 2,
            },
        )

    # ── Assembly lines ──────────────────────────────────

    def _build_assembly_lines(self) -> None:
        lines = [
            ("line_a", "Assembly Line A", "plant_01", "CNC machining and milling"),
            ("line_b", "Assembly Line B", "plant_01", "Welding and fabrication"),
            ("line_c", "Assembly Line C", "plant_02", "Finishing and quality check"),
        ]
        for lid, name, plant_id, desc in lines:
            self._add_entity(
                id=lid,
                name=name,
                entity_type=EntityType.ASSEMBLY_LINE,
                properties={"description": desc, "status": "operational"},
            )
            self._add_rel(lid, plant_id, RelationType.LOCATED_IN)
            self._add_rel(plant_id, lid, RelationType.CONTAINS)

    # ── Machines ────────────────────────────────────────

    def _build_machines(self) -> None:
        machines = [
            # Line A — CNC machines (this is where the RCA story lives)
            {
                "id": "machine_m400",
                "name": "CNC Mill M-400",
                "line": "line_a",
                "props": {
                    "manufacturer": "Haas Automation",
                    "spindle_speed_rpm": 12000,
                    "tolerance_mm": 0.005,
                    "cycle_time_min": 4.5,
                    "certification_required": "Level 3",
                    "status": "degraded",
                    "commissioned": "2021-03-15",
                },
            },
            {
                "id": "machine_m200",
                "name": "CNC Mill M-200",
                "line": "line_a",
                "props": {
                    "manufacturer": "Haas Automation",
                    "spindle_speed_rpm": 10000,
                    "tolerance_mm": 0.01,
                    "cycle_time_min": 3.8,
                    "certification_required": "Level 2",
                    "status": "operational",
                    "commissioned": "2020-06-10",
                },
            },
            {
                "id": "machine_l200",
                "name": "Lathe L-200",
                "line": "line_a",
                "props": {
                    "manufacturer": "DMG Mori",
                    "spindle_speed_rpm": 8000,
                    "tolerance_mm": 0.008,
                    "cycle_time_min": 6.2,
                    "certification_required": "Level 2",
                    "status": "operational",
                    "commissioned": "2022-01-20",
                },
            },
            # Line B — welding
            {
                "id": "machine_w100",
                "name": "Welding Robot W-100",
                "line": "line_b",
                "props": {
                    "manufacturer": "FANUC",
                    "weld_type": "MIG",
                    "max_current_amps": 350,
                    "certification_required": "Level 3",
                    "status": "operational",
                    "commissioned": "2021-09-01",
                },
            },
            {
                "id": "machine_w200",
                "name": "Welding Robot W-200",
                "line": "line_b",
                "props": {
                    "manufacturer": "FANUC",
                    "weld_type": "TIG",
                    "max_current_amps": 250,
                    "certification_required": "Level 3",
                    "status": "operational",
                    "commissioned": "2022-05-12",
                },
            },
            # Line C — finishing
            {
                "id": "machine_g150",
                "name": "Grinder G-150",
                "line": "line_c",
                "props": {
                    "manufacturer": "Studer",
                    "grinding_speed_rpm": 6000,
                    "tolerance_mm": 0.002,
                    "certification_required": "Level 2",
                    "status": "operational",
                    "commissioned": "2023-02-28",
                },
            },
            {
                "id": "machine_p300",
                "name": "Paint Booth P-300",
                "line": "line_c",
                "props": {
                    "manufacturer": "Nordson",
                    "coating_type": "powder",
                    "booth_temp_c": 22,
                    "certification_required": "Level 1",
                    "status": "operational",
                    "commissioned": "2021-11-15",
                },
            },
            {
                "id": "machine_q100",
                "name": "QC Station Q-100",
                "line": "line_c",
                "props": {
                    "manufacturer": "Zeiss",
                    "measurement_type": "CMM",
                    "accuracy_um": 1.5,
                    "certification_required": "Level 3",
                    "status": "operational",
                    "commissioned": "2023-06-01",
                },
            },
        ]
        for m in machines:
            self._add_entity(
                id=m["id"],
                name=m["name"],
                entity_type=EntityType.MACHINE,
                properties=m["props"],
            )
            self._add_rel(m["id"], m["line"], RelationType.LOCATED_IN)
            self._add_rel(m["line"], m["id"], RelationType.CONTAINS)

    # ── Sensors & PLCs ──────────────────────────────────

    def _build_sensors(self) -> None:
        sensor_configs = [
            # M-400 sensors (key for RCA — vibration anomaly here)
            ("sensor_m400_vib", "Vibration Sensor VS-401", "machine_m400", {"metric": "vibration_mm_s", "threshold": 4.5, "current_value": 7.2, "status": "alarm"}),
            ("sensor_m400_temp", "Temp Sensor TS-402", "machine_m400", {"metric": "temperature_c", "threshold": 85, "current_value": 78, "status": "normal"}),
            ("sensor_m400_load", "Spindle Load SL-403", "machine_m400", {"metric": "spindle_load_pct", "threshold": 90, "current_value": 82, "status": "warning"}),
            ("plc_7a", "PLC Controller PLC-7A", "machine_m400", {"monitors": "vibration, temperature, spindle_load", "firmware": "v3.2.1"}),

            # M-200 sensors
            ("sensor_m200_vib", "Vibration Sensor VS-201", "machine_m200", {"metric": "vibration_mm_s", "threshold": 4.5, "current_value": 2.1, "status": "normal"}),
            ("sensor_m200_temp", "Temp Sensor TS-202", "machine_m200", {"metric": "temperature_c", "threshold": 85, "current_value": 65, "status": "normal"}),

            # L-200 sensors (also got bad batch — early warning signs)
            ("sensor_l200_vib", "Vibration Sensor VS-L201", "machine_l200", {"metric": "vibration_mm_s", "threshold": 4.5, "current_value": 3.9, "status": "warning"}),
            ("sensor_l200_temp", "Temp Sensor TS-L202", "machine_l200", {"metric": "temperature_c", "threshold": 85, "current_value": 72, "status": "normal"}),

            # W-100 sensors
            ("sensor_w100_arc", "Arc Monitor AM-101", "machine_w100", {"metric": "arc_stability_pct", "threshold": 95, "current_value": 98, "status": "normal"}),
            ("sensor_w100_temp", "Weld Temp WT-102", "machine_w100", {"metric": "weld_temp_c", "threshold": 1500, "current_value": 1350, "status": "normal"}),

            # G-150 sensors (also got bad batch — trending up)
            ("sensor_g150_vib", "Vibration Sensor VS-G151", "machine_g150", {"metric": "vibration_mm_s", "threshold": 3.0, "current_value": 2.8, "status": "warning"}),
            ("sensor_g150_temp", "Temp Sensor TS-G152", "machine_g150", {"metric": "temperature_c", "threshold": 70, "current_value": 62, "status": "normal"}),
        ]
        for sid, name, machine_id, props in sensor_configs:
            etype = EntityType.PLC if sid.startswith("plc") else EntityType.SENSOR
            self._add_entity(id=sid, name=name, entity_type=etype, properties=props)
            self._add_rel(machine_id, sid, RelationType.MONITORED_BY)
            self._add_rel(sid, machine_id, RelationType.READS_FROM)

    # ── Personnel ───────────────────────────────────────

    def _build_personnel(self) -> None:
        technicians = [
            ("tech_ravi", "Ravi Kumar", EntityType.TECHNICIAN, {"specialization": "CNC maintenance", "certification": "Level 3", "shift": "morning", "years_experience": 8}),
            ("tech_priya", "Priya Sharma", EntityType.TECHNICIAN, {"specialization": "Welding systems", "certification": "Level 3", "shift": "afternoon", "years_experience": 5}),
            ("tech_arun", "Arun Patel", EntityType.TECHNICIAN, {"specialization": "General maintenance", "certification": "Level 2", "shift": "morning", "years_experience": 3}),
        ]
        operators = [
            ("op_suresh", "Suresh Reddy", EntityType.OPERATOR, {"line": "line_a", "certification": "Level 3", "shift": "morning"}),
            ("op_meena", "Meena Devi", EntityType.OPERATOR, {"line": "line_b", "certification": "Level 3", "shift": "morning"}),
            ("op_vijay", "Vijay Singh", EntityType.OPERATOR, {"line": "line_c", "certification": "Level 2", "shift": "afternoon"}),
        ]
        for pid, name, etype, props in technicians + operators:
            self._add_entity(id=pid, name=name, entity_type=etype, properties=props)

        # Operator → machine assignments
        op_assignments = [
            ("op_suresh", "machine_m400"),
            ("op_suresh", "machine_m200"),
            ("op_suresh", "machine_l200"),
            ("op_meena", "machine_w100"),
            ("op_meena", "machine_w200"),
            ("op_vijay", "machine_g150"),
            ("op_vijay", "machine_p300"),
            ("op_vijay", "machine_q100"),
        ]
        for op_id, machine_id in op_assignments:
            self._add_rel(op_id, machine_id, RelationType.OPERATES)

    # ── Suppliers & batches ─────────────────────────────

    def _build_suppliers_and_batches(self) -> None:
        suppliers = [
            ("supplier_precision", "Precision Parts Co.", {"location": "Chennai", "rating": "A", "contract_since": "2019"}),
            ("supplier_steelmax", "SteelMax Industries", {"location": "Mumbai", "rating": "A+", "contract_since": "2018"}),
            ("supplier_bearingworld", "BearingWorld Ltd.", {"location": "Pune", "rating": "B+", "contract_since": "2021"}),
        ]
        for sid, name, props in suppliers:
            self._add_entity(id=sid, name=name, entity_type=EntityType.SUPPLIER, properties=props)

        # THE BAD BATCH — B-442 from Precision Parts Co.
        batches = [
            ("batch_b442", "Batch B-442", "supplier_precision", {"manufactured": "2024-12-01", "quantity": 50, "part_type": "spindle_bearing", "quality_cert": "ISO-9001", "status": "under_investigation"}),
            ("batch_b440", "Batch B-440", "supplier_precision", {"manufactured": "2024-11-15", "quantity": 50, "part_type": "spindle_bearing", "quality_cert": "ISO-9001", "status": "cleared"}),
            ("batch_s100", "Batch S-100", "supplier_steelmax", {"manufactured": "2024-11-20", "quantity": 200, "part_type": "steel_rod_304", "quality_cert": "ISO-9001", "status": "cleared"}),
            ("batch_bw220", "Batch BW-220", "supplier_bearingworld", {"manufactured": "2024-12-10", "quantity": 30, "part_type": "linear_bearing", "quality_cert": "ISO-9001", "status": "cleared"}),
        ]
        for bid, name, supplier_id, props in batches:
            self._add_entity(id=bid, name=name, entity_type=EntityType.SUPPLIER_BATCH, properties=props)
            self._add_rel(bid, supplier_id, RelationType.BATCH_OWNED_BY)

    # ── Parts ───────────────────────────────────────────

    def _build_parts(self) -> None:
        parts = [
            # Bad batch B-442 parts → installed in 3 machines (the RCA chain)
            ("part_bearing_m400", "Spindle Bearing #M400-SB", "batch_b442", "machine_m400", {"part_type": "spindle_bearing", "installed_date": "2025-01-15", "expected_life_hours": 5000}),
            ("part_bearing_l200", "Spindle Bearing #L200-SB", "batch_b442", "machine_l200", {"part_type": "spindle_bearing", "installed_date": "2025-01-18", "expected_life_hours": 5000}),
            ("part_bearing_g150", "Spindle Bearing #G150-SB", "batch_b442", "machine_g150", {"part_type": "spindle_bearing", "installed_date": "2025-01-20", "expected_life_hours": 5000}),

            # Good batch parts
            ("part_bearing_m200", "Spindle Bearing #M200-SB", "batch_b440", "machine_m200", {"part_type": "spindle_bearing", "installed_date": "2024-12-10", "expected_life_hours": 5000}),
            ("part_rod_w100", "Steel Rod #W100-SR", "batch_s100", "machine_w100", {"part_type": "steel_rod", "installed_date": "2024-12-15", "expected_life_hours": 8000}),
            ("part_lbearing_q100", "Linear Bearing #Q100-LB", "batch_bw220", "machine_q100", {"part_type": "linear_bearing", "installed_date": "2025-01-05", "expected_life_hours": 6000}),
        ]
        for pid, name, batch_id, machine_id, props in parts:
            self._add_entity(id=pid, name=name, entity_type=EntityType.PART, properties=props)
            self._add_rel(pid, batch_id, RelationType.FROM_BATCH)
            self._add_rel(pid, machine_id, RelationType.INSTALLED_IN)
            # Find the supplier from the batch
            batch_supplier_rels = [
                r for r in self._relationships
                if r.source_id == batch_id and r.relation_type == RelationType.BATCH_OWNED_BY
            ]
            if batch_supplier_rels:
                self._add_rel(pid, batch_supplier_rels[0].target_id, RelationType.SUPPLIED_BY)

    # ── Maintenance events ──────────────────────────────

    def _build_maintenance_events(self) -> None:
        base_date = datetime(2025, 1, 15)
        events = [
            # THE KEY EVENT — Ravi replaces bearing on M-400 with bad batch part
            {
                "id": "maint_m400_jan15",
                "name": "M-400 Bearing Replacement",
                "machine": "machine_m400",
                "tech": "tech_ravi",
                "type": MaintenanceType.PREVENTIVE,
                "date": base_date,
                "desc": "Scheduled spindle bearing replacement. Old bearing showed normal wear. Replaced with new bearing from Batch B-442.",
                "parts": ["part_bearing_m400"],
            },
            # L-200 also got a bearing from B-442
            {
                "id": "maint_l200_jan18",
                "name": "L-200 Bearing Replacement",
                "machine": "machine_l200",
                "tech": "tech_ravi",
                "type": MaintenanceType.PREVENTIVE,
                "date": base_date + timedelta(days=3),
                "desc": "Scheduled spindle bearing replacement on Lathe L-200. Replaced with bearing from Batch B-442.",
                "parts": ["part_bearing_l200"],
            },
            # G-150 also got one
            {
                "id": "maint_g150_jan20",
                "name": "G-150 Bearing Replacement",
                "machine": "machine_g150",
                "tech": "tech_arun",
                "type": MaintenanceType.PREVENTIVE,
                "date": base_date + timedelta(days=5),
                "desc": "Scheduled bearing replacement on Grinder G-150. Used bearing from Batch B-442.",
                "parts": ["part_bearing_g150"],
            },
            # Unrelated maintenance — good batch
            {
                "id": "maint_m200_dec10",
                "name": "M-200 Bearing Replacement",
                "machine": "machine_m200",
                "tech": "tech_ravi",
                "type": MaintenanceType.PREVENTIVE,
                "date": datetime(2024, 12, 10),
                "desc": "Routine bearing replacement on M-200. Used bearing from Batch B-440. No issues.",
                "parts": ["part_bearing_m200"],
            },
            # Welding robot calibration
            {
                "id": "maint_w100_jan10",
                "name": "W-100 Arc Calibration",
                "machine": "machine_w100",
                "tech": "tech_priya",
                "type": MaintenanceType.CALIBRATION,
                "date": base_date - timedelta(days=5),
                "desc": "Quarterly arc calibration on Welding Robot W-100. All parameters within spec.",
                "parts": [],
            },
        ]
        for evt in events:
            self._add_entity(
                id=evt["id"],
                name=evt["name"],
                entity_type=EntityType.MAINTENANCE_EVENT,
                properties={
                    "maintenance_type": evt["type"].value,
                    "date": evt["date"].isoformat(),
                    "description": evt["desc"],
                },
            )
            self._add_rel(evt["id"], evt["machine"], RelationType.PERFORMED_ON)
            self._add_rel(evt["id"], evt["tech"], RelationType.PERFORMED_BY)
            for part_id in evt["parts"]:
                self._add_rel(evt["id"], part_id, RelationType.REPLACED_PART)

    # ── Alerts & defects ────────────────────────────────

    def _build_alerts_and_defects(self) -> None:
        # Vibration alert on M-400 (triggered by bad bearing)
        self._add_entity(
            id="alert_m400_vib",
            name="M-400 Vibration Alert",
            entity_type=EntityType.ALERT,
            properties={
                "severity": AlertSeverity.CRITICAL.value,
                "triggered_at": "2025-02-10T08:30:00",
                "metric": "vibration_mm_s",
                "value": 7.2,
                "threshold": 4.5,
                "message": "Spindle vibration exceeded critical threshold on CNC Mill M-400",
            },
        )
        self._add_rel("sensor_m400_vib", "alert_m400_vib", RelationType.TRIGGERED_ALERT)
        self._add_rel("alert_m400_vib", "machine_m400", RelationType.ALERT_FOR)

        # Warning on L-200 (early sign — same bad batch)
        self._add_entity(
            id="alert_l200_vib",
            name="L-200 Vibration Warning",
            entity_type=EntityType.ALERT,
            properties={
                "severity": AlertSeverity.WARNING.value,
                "triggered_at": "2025-02-12T14:15:00",
                "metric": "vibration_mm_s",
                "value": 3.9,
                "threshold": 4.5,
                "message": "Spindle vibration trending upward on Lathe L-200. Approaching threshold.",
            },
        )
        self._add_rel("sensor_l200_vib", "alert_l200_vib", RelationType.TRIGGERED_ALERT)
        self._add_rel("alert_l200_vib", "machine_l200", RelationType.ALERT_FOR)

        # Warning on G-150 (early sign — same bad batch)
        self._add_entity(
            id="alert_g150_vib",
            name="G-150 Vibration Warning",
            entity_type=EntityType.ALERT,
            properties={
                "severity": AlertSeverity.WARNING.value,
                "triggered_at": "2025-02-13T09:45:00",
                "metric": "vibration_mm_s",
                "value": 2.8,
                "threshold": 3.0,
                "message": "Vibration trending upward on Grinder G-150. Monitor closely.",
            },
        )
        self._add_rel("sensor_g150_vib", "alert_g150_vib", RelationType.TRIGGERED_ALERT)
        self._add_rel("alert_g150_vib", "machine_g150", RelationType.ALERT_FOR)

        # Defect report — the thing the operator notices
        self._add_entity(
            id="defect_001",
            name="Surface Finish Defect D-001",
            entity_type=EntityType.DEFECT,
            properties={
                "severity": AlertSeverity.CRITICAL.value,
                "detected_at": "2025-02-10T10:00:00",
                "description": "Parts from CNC Mill M-400 showing surface roughness outside tolerance. Ra value 3.2μm vs spec 1.6μm.",
                "affected_parts_count": 47,
                "shift": "morning",
                "detected_by": "op_suresh",
            },
        )
        self._add_rel("defect_001", "machine_m400", RelationType.DEFECT_ON)

    # ── Materials ───────────────────────────────────────

    def _build_materials(self) -> None:
        materials = [
            ("mat_al6061", "Aluminum Alloy 6061", {"grade": "T6", "hardness_hrc": 40, "use": "structural parts"}),
            ("mat_ss304", "Stainless Steel 304", {"grade": "304L", "hardness_hrc": 70, "use": "corrosion-resistant parts"}),
            ("mat_ti64", "Titanium Ti-6Al-4V", {"grade": "Grade 5", "hardness_hrc": 36, "use": "aerospace components"}),
        ]
        for mid, name, props in materials:
            self._add_entity(id=mid, name=name, entity_type=EntityType.MATERIAL, properties=props)

        # Machine → material relationships
        machine_materials = [
            ("machine_m400", "mat_al6061"),
            ("machine_m400", "mat_ss304"),
            ("machine_m200", "mat_al6061"),
            ("machine_l200", "mat_ss304"),
            ("machine_l200", "mat_ti64"),
            ("machine_g150", "mat_ss304"),
        ]
        for machine_id, mat_id in machine_materials:
            self._add_rel(machine_id, mat_id, RelationType.PROCESSES)

    # ── Process logs ────────────────────────────────────

    def _build_process_logs(self) -> None:
        logs = [
            ("log_m400_0210", "M-400 Process Log Feb 10", "machine_m400", {
                "date": "2025-02-10",
                "shift": "morning",
                "operator": "op_suresh",
                "parts_produced": 85,
                "parts_rejected": 47,
                "rejection_rate_pct": 55.3,
                "notes": "High rejection rate. Operator reported unusual vibration. Production halted at 10:15 AM.",
            }),
            ("log_m400_0209", "M-400 Process Log Feb 9", "machine_m400", {
                "date": "2025-02-09",
                "shift": "morning",
                "operator": "op_suresh",
                "parts_produced": 120,
                "parts_rejected": 8,
                "rejection_rate_pct": 6.7,
                "notes": "Slightly elevated rejections. Vibration readings noted by operator.",
            }),
            ("log_m400_0208", "M-400 Process Log Feb 8", "machine_m400", {
                "date": "2025-02-08",
                "shift": "morning",
                "operator": "op_suresh",
                "parts_produced": 130,
                "parts_rejected": 2,
                "rejection_rate_pct": 1.5,
                "notes": "Normal operations.",
            }),
            ("log_l200_0212", "L-200 Process Log Feb 12", "machine_l200", {
                "date": "2025-02-12",
                "shift": "morning",
                "operator": "op_suresh",
                "parts_produced": 95,
                "parts_rejected": 5,
                "rejection_rate_pct": 5.3,
                "notes": "Minor uptick in rejections. Vibration sensor showing upward trend.",
            }),
        ]
        for lid, name, machine_id, props in logs:
            self._add_entity(
                id=lid,
                name=name,
                entity_type=EntityType.PROCESS_LOG,
                properties=props,
            )
            self._add_rel(lid, machine_id, RelationType.LOGGED_BY)