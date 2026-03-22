"""
Domain enumerations for the Graph RAG demo.
Defines entity types, relationship types, and RAG modes
used across the manufacturing knowledge graph.
"""

from enum import Enum


class RAGMode(str, Enum):
    """Which RAG pipeline to use."""
    BASIC = "basic"
    CONTEXT = "context"


class EntityType(str, Enum):
    """Node types in the manufacturing knowledge graph."""
    PLANT = "plant"
    ASSEMBLY_LINE = "assembly_line"
    MACHINE = "machine"
    SENSOR = "sensor"
    PLC = "plc"
    MAINTENANCE_EVENT = "maintenance_event"
    TECHNICIAN = "technician"
    OPERATOR = "operator"
    PART = "part"
    SUPPLIER = "supplier"
    SUPPLIER_BATCH = "supplier_batch"
    DEFECT = "defect"
    ALERT = "alert"
    MATERIAL = "material"
    PROCESS_LOG = "process_log"


class RelationType(str, Enum):
    """Edge types in the manufacturing knowledge graph."""

    # Structural / spatial
    LOCATED_IN = "located_in"           # machine → assembly_line → plant
    CONTAINS = "contains"               # plant → assembly_line, line → machine

    # Monitoring
    MONITORED_BY = "monitored_by"       # machine → sensor / plc
    READS_FROM = "reads_from"           # sensor → machine (metric readings)

    # Maintenance
    PERFORMED_ON = "performed_on"       # maintenance_event → machine
    PERFORMED_BY = "performed_by"       # maintenance_event → technician
    REPLACED_PART = "replaced_part"     # maintenance_event → part

    # Supply chain
    SUPPLIED_BY = "supplied_by"         # part → supplier
    FROM_BATCH = "from_batch"           # part → supplier_batch
    BATCH_OWNED_BY = "batch_owned_by"   # supplier_batch → supplier
    INSTALLED_IN = "installed_in"       # part → machine

    # Incidents / RCA
    TRIGGERED_ALERT = "triggered_alert" # sensor reading → alert
    ALERT_FOR = "alert_for"             # alert → machine
    CAUSED_DEFECT = "caused_defect"     # root cause → defect
    DEFECT_ON = "defect_on"             # defect → machine

    # Operations
    OPERATES = "operates"               # operator → machine
    PROCESSES = "processes"             # machine → material
    LOGGED_BY = "logged_by"             # process_log → machine


class ContextEdgeType(str, Enum):
    """
    Edge types specific to the session context graph.
    These are created dynamically during conversation,
    NOT part of the static knowledge graph.
    """
    DISCUSSED = "discussed"             # user discussed this entity
    RESOLVED_TO = "resolved_to"         # coreference: "it" → Machine M-400
    COMPARED_WITH = "compared_with"     # user compared two entities
    SUSPECTED_CAUSE = "suspected_cause" # RCA: entity suspected as root cause
    RULED_OUT = "ruled_out"             # RCA: entity eliminated from investigation
    FOLLOW_UP = "follow_up"             # entity needs further investigation


class AlertSeverity(str, Enum):
    """Severity levels for machine alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MaintenanceType(str, Enum):
    """Types of maintenance events."""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"
    CALIBRATION = "calibration"