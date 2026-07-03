from __future__ import annotations

from .utils import (
    build_impact_summary,
    describe_change_ratio,
    describe_region,
    extract_incident_info,
    get_priority_label,
)


def summarize_retrieved_guidance(chunks):
    """Summarize retrieved guidance exactly as in the notebook."""
    if not chunks:
        return (
            [
                "Conduct detailed structural inspections before allowing re-entry.",
                "Restrict access to unsafe or unstable buildings.",
                "Prioritize emergency response in the most severely affected areas.",
                "Restore critical utilities only after structural safety is confirmed.",
            ],
            [],
        )

    recommendations = []
    source_files = []

    def add(rec):
        if rec not in recommendations:
            recommendations.append(rec)

    for chunk in chunks:
        text = chunk["chunk_text"].lower()
        source_files.append(chunk["file_name"])

        if "inspect" in text or "inspection" in text or "assessment" in text or "evaluate" in text:
            add("Conduct detailed structural inspections before allowing re-occupancy.")

        if "unsafe" in text or "restrict" in text or "red tag" in text or "access" in text:
            add("Restrict access to buildings identified as structurally unsafe.")

        if "evac" in text or "life safety" in text or "occupancy" in text:
            add("Evacuate occupants from structures presenting immediate safety risks.")

        if "search and rescue" in text or "rescue" in text:
            add("Prioritize search and rescue operations where occupancy is uncertain.")

        if "stabil" in text or "temporary support" in text:
            add("Stabilize damaged structures before initiating recovery activities.")

        if "debris" in text or "clearance" in text or "demolition" in text:
            add("Begin debris clearance only after completing structural safety assessments.")

        if (
            "utility" in text
            or "lifeline" in text
            or "electric" in text
            or "water" in text
            or "gas" in text
        ):
            add("Restore essential utilities after confirming infrastructure safety.")

        if "recovery" in text or "repair" in text or "rehabilitation" in text:
            add("Prioritize repair of buildings that remain structurally serviceable.")

        if "monitor" in text or "aftershock" in text or "secondary hazard" in text:
            add("Continue monitoring damaged structures for secondary hazards.")

    if len(recommendations) == 0:
        recommendations = [
            "Conduct structural inspections before allowing re-entry.",
            "Restrict access to unsafe structures.",
            "Prioritize emergency response in affected areas.",
            "Restore essential services following safety verification.",
        ]

    recommendations = recommendations[:5]
    source_files = sorted(set(source_files))

    return recommendations, source_files


def generate_report_text(row, retrieved_chunks):
    """Generate the notebook's final report text."""
    affected_pct = row["pred_affected_ratio"] * 100

    destroyed = int(row["pred_destroyed"])
    damaged = int(row["pred_damaged"])
    no_damage = int(row["pred_no_damage"])
    total = int(row["total_buildings"])

    disaster_type, event_name = extract_incident_info(row["image_id"])
    priority = get_priority_label(row)
    recommendations, source_files = summarize_retrieved_guidance(retrieved_chunks)

    if destroyed >= max(5, 0.20 * total):
        interpretation = ("Extensive structural destruction has been identified across the assessment area, indicating a high-risk environment requiring immediate emergency response, restricted access and detailed engineering assessment.")
    
    elif affected_pct >= 60:
        interpretation = ("A large proportion of assessed buildings exhibit visible structural damage. Although complete collapse is limited, the overall impact suggests substantial disruption requiring coordinated inspection and recovery operations.")
    
    elif affected_pct >= 25:
        interpretation = ("Damage appears moderate and concentrated within portions of the scene. Field verification is recommended before initiating large-scale recovery activities.")
    
    else:
        interpretation = ("Only limited structural damage is evident. Localized inspections should be conducted to verify building safety.")

    report = f"""
\033[1mPOST-DISASTER DAMAGE ASSESSMENT\033[0m

\033[1mIncident Information\033[0m

Event: {event_name}

\033[1mAssessment Metadata\033[0m

Buildings Identified: {total}
Assessment Type: AI-based Building Damage Assessment

Priority Level: {priority}

\033[1mSituation Summary\033[0m

A total of {total} buildings were identified within the assessment area.

• Destroyed: {destroyed}
• Damaged: {damaged}
• No Damage: {no_damage}

Overall affected buildings: {affected_pct:.1f}%.

\033[1mImpact Assessment\033[0m

{build_impact_summary(row)}

\033[1mDamage Interpretation\033[0m

{interpretation}

\033[1mSpatial Analysis\033[0m

{describe_change_ratio(row['changed_area_ratio'])}

The highest concentration of visible damage is located in the {describe_region(row['dominant_region'])}.

\033[1mOperational Recommendations\033[0m
"""

    for rec in recommendations:
        report += f"\n• {rec}"

    report += """

\033[1mAssessment Note\033[0m

This report has been generated automatically using satellite imagery, deep learning-based building damage assessment and retrieval-augmented guidance from disaster management documents. The assessment is intended to support rapid situational awareness and should be verified through field inspection before operational decision-making.
"""

    return report
