"""Training-only rendering of generated LOCA evaluator reference artifacts.

Called only after NativeLOCAEnv has checked split membership. No reference is
added to actor observations, MCP mounts, public manifests or evaluation logs.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
import re
from typing import Any


# The evaluator's immutable generated inputs, not the actor-mutated databases.
# Dynamic tasks have no standalone answer file and therefore need explicit
# template-specific inputs in addition to their target-state contract.
_REFERENCE_FILES = {
    "CanvasArrangeExamS2LEnv": ("groundtruth_workspace/exam_schedule.xlsx", "groundtruth_workspace/metadata.json"),
    "CanvasListTestS2LEnv": ("groundtruth_workspace/quiz_info.csv", "groundtruth_workspace/assignment_info.csv"),
    "CourseAssistantS2LEnv": ("initial_workspace/nlp_statistics.xlsx", "files/emails.jsonl"),
    "ABTestingS2LEnv": ("groundtruth_workspace/expected_ratio.csv",),
    "AcademicWarningS2LEnv": ("groundtruth_workspace/expected_alerts.csv",),
    "MachineOperatingS2LEnv": ("groundtruth_workspace/anomaly_report.csv",),
    "SetConfCrDdlS2LEnv": ("groundtruth_workspace/conference_metadata.json", "groundtruth_workspace/today.txt"),
    "ExcelMarketResearchS2LEnv": ("groundtruth_workspace/Market_Data_gt.csv", "groundtruth_workspace/metadata.json"),
    "FilterLowSellingProductsS2LEnv": ("groundtruth_workspace/generation_metadata.json", "preprocess/generated_products.json", "initial_workspace/subscriber.json"),
    "UpdateMaterialInventoryS2LEnv": ("groundtruth_workspace/expected_results.json", "groundtruth_workspace/config.json"),
    "WoocommerceNewWelcomeS2LEnv": ("preprocess/customers_data.json", "preprocess/generated_orders.json", "initial_workspace/welcome_email_template.md"),
    "WoocommerceStockAlertS2LEnv": ("preprocess/woocommerce_products.json", "files/sheet_id.txt", "initial_workspace/stock_alert_email_template.md", "initial_workspace/purchasing_manager_email.txt"),
    "NhlB2bAnalysisS2LEnv": ("groundtruth_workspace/standard_answer.csv", "groundtruth_workspace/generation_metadata.json"),
    "PayableInvoiceCheckerS2LEnv": ("groundtruth_workspace/invoice.jsonl", "groundtruth_workspace/interference_invoices.jsonl", "groundtruth_workspace/interference_payments.jsonl"),
}

_TARGET_CONTRACTS = {
    "CanvasArrangeExamS2LEnv": "Create exam_schedule.xlsx with every reference row and all 11 columns, sorted chronologically with TBD entries last.",
    "CanvasListTestS2LEnv": "Create both quiz_info.csv and assignment_info.csv with all reference rows, columns and required sorting/normalization.",
    "CourseAssistantS2LEnv": "Each non-dropped student whose ID has no nlp-presentation submission must receive exactly one nlp-course-emergency email containing their name and ID. Submitted and dropped students receive none.",
    "ABTestingS2LEnv": "Write record.csv matching every A/B scenario percentage (0.1 percentage-point tolerance). If overall B>A, create promo-assets-for-b and keep abtesting_logging clean. Otherwise do not create that bucket and log status=AB_Test_Concluded,winner=A,action=No_Change.",
    "AcademicWarningS2LEnv": "bad_student.csv must contain exactly the expected students (drop_ratio>0.25). Every student with drop_ratio>0.45 requires an exam_log CRITICAL message containing that student's ID and name together; all other students must have no matching CRITICAL message.",
    "MachineOperatingS2LEnv": "Upload an anomaly_report-named file to bucket iot_anomaly_reports, matching every reference timestamp/machine_id/sensor_type/reading in both directions (time tolerance 60 seconds, reading tolerance 0.01).",
    "SetConfCrDdlS2LEnv": "For every target_info.conferences item create a calendar reminder. Summary contains conference abbreviation or full name and camera/ready; start is its camera-ready deadline minus 3 hours, with the original timezone (300-second tolerance).",
    "ExcelMarketResearchS2LEnv": "Create growth_rate.xlsx matching reference growth rates by Year, using metadata's dynamic category names and ignoring columns with zero weight; tolerance is one percentage point.",
    "FilterLowSellingProductsS2LEnv": "Select all products with days_in_stock>90 and sales_30_days<10, sorted by descending days then discount_rate=(regular-sale)/regular. Include original/sale prices and discount percentages in CSV, place exactly these products in Outlet/Clearance, and email the correct CSV to every subscriber. Price tolerance 0.01; discount tolerance 0.2 percentage point. Fulfil the task's blog request too; the official evaluator skips that blog check.",
    "UpdateMaterialInventoryS2LEnv": "Match expected_final_inventories exactly for WooCommerce SKUs and Material_Inventory Google Sheet materials (material tolerance 0.01). Product availability is floor(min(remaining material / BOM unit requirement)), as computed in the reference.",
    "WoocommerceNewWelcomeS2LEnv": "Preserve all initial customers and their identity/phone fields. Deduplicate first_time_orders by email; insert these customers into BigQuery with welcome_email_sent=true and nonempty date. Do not mark unsent historical or unknown customers as newly welcomed. Send welcome messages from admin@woocommerce.local only to first-time customers, matching the full provided template and subject Welcome to ... Exclusive offers await you.",
    "WoocommerceStockAlertS2LEnv": "Write exactly the low-stock products (stock_quantity<stock_threshold) to Stock Alert, with Product ID/Product Name/SKU/Current Stock/Safety Threshold/Supplier Name/Supplier ID/Supplier Contact. Exclude normal-stock SKUs. Send at least one Stock Alert email per target SKU to laura_thompson@mcp.com, covering all SKUs, mentioning purchasing manager/stock/threshold, and replacing google_sheets_link with the real sheet link.",
    "NhlB2bAnalysisS2LEnv": "Produce both the local analysis CSV and Google Sheet titled nhl_b2b_analysis. Team/HA/AH/HH/AA/Total must agree with the reference and with each other.",
    "PayableInvoiceCheckerS2LEnv": "Preserve target and interference invoice/payment rows, update payment flags/paid_amount (0.01 tolerance), and set OUTSTANDING_FLAG comment to 0=Paid, 1=Outstanding. Group flag==1 invoice_id.pdf names by buyer_email: exactly one Process outstanding invoices email per unpaid buyer listing all their files. Paid-only buyers and interference recipients receive none.",
}


def _cell_value(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return {"type": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, timedelta):
        return {"type": "timedelta", "seconds": value.total_seconds()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"Unsupported reference spreadsheet value type: {type(value).__name__}")


def read_training_reference(directory: Path, template: str, *, max_characters: int = 131072) -> dict[str, Any]:
    """Return complete readable file contents, failing explicitly on truncation."""
    if not directory.is_dir():
        raise ValueError("The generated LOCA task has no private reference input directory.")
    root = directory.resolve()
    if template == "ApplyPhDEmailS2LEnv":
        return _phd_reference(root, max_characters)
    if template not in _REFERENCE_FILES:
        raise ValueError(f"LOCA template needs an explicit training reference contract: {template}")
    selected = [root / relative for relative in _REFERENCE_FILES[template]]
    if template == "AcademicWarningS2LEnv":
        scores = sorted((root / "files").glob("scores_*.csv"))
        if not scores:
            raise ValueError("AcademicWarning reference lacks the evaluator's initial student roster.")
        selected.append(scores[0])
    artifacts = []
    for path in selected:
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError("LOCA reference contains a path outside its private directory.")
        if not path.is_file():
            raise ValueError(f"LOCA evaluator reference input is missing: {path.relative_to(root)}")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError(f"LOCA reference artifact exceeds the bounded reader: {path.name}")
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv", ".json", ".jsonl", ".txt", ".md", ".yaml", ".yml"}:
            content: Any = path.read_text(encoding="utf-8")
        elif suffix == ".xlsx":
            from openpyxl import load_workbook
            workbook = load_workbook(path, read_only=True, data_only=False)
            try:
                content = {sheet.title: [
                    {"coordinate": cell.coordinate, "type": cell.data_type,
                     "value": _cell_value(cell.value), "number_format": cell.number_format}
                    for row in sheet.iter_rows() for cell in row if cell.value is not None
                ] for sheet in workbook.worksheets}
            finally:
                workbook.close()
        else:
            raise ValueError(f"LOCA reference format needs a faithful reader: {path.name}")
        artifacts.append({"path": path.relative_to(root).as_posix(), "format": suffix.lstrip("."),
                          "content": content})
        if len(json.dumps(artifacts, ensure_ascii=False, allow_nan=False)) > max_characters:
            raise ValueError("Complete LOCA training reference exceeds the teacher reference bound; no content was truncated.")
    if not artifacts:
        raise ValueError("The generated LOCA task has no readable reference artifacts.")
    reference = {"kind": "official_evaluator_inputs_and_target_state", "files": artifacts,
                 "target_state_contract": _TARGET_CONTRACTS[template],
                 "scope": "Training-only target-state supervision; the official benchmark provides no gold action trajectory."}
    if template == "CourseAssistantS2LEnv":
        from openpyxl import load_workbook
        submitted = set()
        for line in (root / "files/emails.jsonl").read_text().splitlines():
            try:
                message = json.loads(line)
                match = re.search(r'nlp-presentation-(\d+)-', message.get("subject", ""))
            except (ValueError, TypeError, AttributeError):
                continue
            if match:
                submitted.add(match.group(1))
        book = load_workbook(root / "initial_workspace/nlp_statistics.xlsx", read_only=True)
        try:
            students = [{"name": row[0], "student_id": str(row[1]), "email": row[2], "status": row[3]}
                        for row in book.active.iter_rows(min_row=2, values_only=True) if row[0]]
        finally:
            book.close()
        reference["required_recipients"] = [s for s in students if s["status"] != "dropped" and s["student_id"] not in submitted]
        reference["forbidden_recipients"] = [s for s in students if s["status"] == "dropped" or s["student_id"] in submitted]
    elif template == "WoocommerceStockAlertS2LEnv":
        products = json.loads((root / "preprocess/woocommerce_products.json").read_text())["products"]
        reference["expected_products"] = [p for p in products if p.get("stock_quantity", 0) < p.get("stock_threshold", 0)]
        reference["forbidden_products"] = [p for p in products if p.get("stock_quantity", 0) >= p.get("stock_threshold", 0)]
    elif template == "WoocommerceNewWelcomeS2LEnv":
        orders = json.loads((root / "preprocess/generated_orders.json").read_text())["first_time_orders"]
        seen, customers = set(), []
        for order in orders:
            email = order.get("customer_email") or order.get("billing", {}).get("email")
            if email and email not in seen:
                seen.add(email)
                customers.append(order)
        reference["first_time_customer_orders"] = customers
    elif template == "PayableInvoiceCheckerS2LEnv":
        reference["additional_forbidden_recipients"] = [
            "JSmith@mcp.com", "MBrown@mcp.com", "AWilliams@mcp.com", "RJohnson@mcp.com",
            "LDavis@mcp.com", "KWilson@mcp.com", "TMiller@mcp.com", "SAnderson@mcp.com",
        ]
    _check_bound(reference, max_characters)
    return reference


def _check_bound(reference: Any, max_characters: int) -> None:
    if len(json.dumps(reference, ensure_ascii=False, allow_nan=False)) > max_characters:
        raise ValueError("Complete LOCA training reference exceeds the teacher reference bound; no content was truncated.")


def _identity_only(value: Any) -> Any:
    if isinstance(value, list):
        return [_identity_only(item) for item in value]
    if isinstance(value, dict):
        return {key: _identity_only(item) for key, item in value.items()
                if not any(word in key.lower() for word in ("password", "token", "secret", "api_key"))}
    return value


def _phd_reference(root: Path, max_characters: int) -> dict[str, Any]:
    config = json.loads((root / "task_config_generated.json").read_text())
    if not config.get("assign_different_structures"):
        raise ValueError("ApplyPhD reference contract covers the registered per-professor structure setting only.")
    positive = config.get("positive_professors")
    if not positive:
        raise ValueError("ApplyPhD reference has no generated positive professors.")
    structures = config.get("structure_info", {})
    if any(not structures.get(professor["email"], {}).get("structure_info") for professor in positive):
        raise ValueError("ApplyPhD reference lacks a complete per-professor material tree.")
    reference = {
        "kind": "official_evaluator_inputs_and_target_state",
        "task_configuration": _identity_only(config),
        "email_identities": _identity_only(json.loads((root / "email_config.json").read_text())),
        "receiver_identities": _identity_only(json.loads((root / "files/receiver_config.json").read_text())),
        "target_state_contract": (
            "Send each positive professor their own configured materials ZIP, with subject containing submit_material. "
            "Preserve that professor's complete structure_info folders/files under an Application_Materials_* root. "
            "Recommendation_Letter_[ProfessorName]-1 maps to Alex-1 and -2 maps to Lily-2 (ProfessorAlex/ProfessorLily aliases allowed). "
            "When All_Awards_Certificates.pdf is required, merge the three original certificates in this order: "
            "Outstanding Student Award 2021; Research Competition First Place 2022; Academic Excellence Award 2023. "
            "Use the real supplied materials and photo; do not replace them with empty or synthetic placeholders."
        ),
        "scope": "Training-only target-state supervision; binary source materials remain native task inputs.",
    }
    _check_bound(reference, max_characters)
    return reference
