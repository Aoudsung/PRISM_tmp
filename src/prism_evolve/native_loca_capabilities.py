"""Explicit public tool partition for PoS LOCA 8k, pinned deployment versions.

I means information intent and includes native cache/search-cursor side effects.
E includes arbitrary code and mixed SQL execution. The classification is made
before execution, without reading task labels, permissions or results. Names are
the source registrations before FastMCP adds the configured server prefix.

Sources: PoS 6818cfa, bundled LOCA 8b6fac4; filesystem 2.2.0; memory 2.1.0;
excel-mcp-server 0.1.8; cli-mcp-server 0.2.5; pdf-tools-mcp 0.1.4.
"""
from __future__ import annotations

from typing import Mapping


_CATALOG = {
    "calendar": (
        "get_event list_events",
        "create_event update_event delete_event",
    ),
    "email": (
        "get_current_user list_users check_connection get_folders get_emails read_email search_emails "
        "get_drafts get_mailbox_stats get_unread_count get_email_headers download_attachment",
        "login logout create_user create_folder delete_folder send_email reply_email forward_email "
        "delete_email move_email mark_emails move_emails delete_emails save_draft update_draft "
        "delete_draft export_emails import_emails",
    ),
    "google_sheet": (
        "get_sheet_data get_sheet_formulas list_sheets get_multiple_sheet_data "
        "get_multiple_spreadsheet_summary list_spreadsheets",
        "update_cells batch_update_cells add_rows add_columns copy_sheet rename_sheet "
        "create_spreadsheet create_sheet share_spreadsheet",
    ),
    "snowflake": (
        "list_databases list_schemas list_tables describe_table read_query",
        "write_query create_table append_insight",
    ),
    "google_cloud": (
        "bigquery_list_datasets bigquery_get_dataset_info bigquery_list_jobs storage_list_buckets "
        "storage_list_objects storage_get_bucket_info storage_generate_signed_url storage_get_bucket_size "
        "compute_list_instances compute_get_instance compute_list_zones compute_wait_for_operation "
        "logging_read_logs logging_list_logs logging_list_log_buckets logging_list_log_sinks",
        "bigquery_run_query bigquery_create_dataset bigquery_load_csv_data bigquery_export_table "
        "bigquery_cancel_job storage_create_bucket storage_upload_file storage_download_file "
        "storage_delete_object storage_copy_object storage_move_object storage_enable_versioning "
        "storage_set_bucket_lifecycle compute_create_instance compute_delete_instance compute_start_instance "
        "compute_stop_instance compute_restart_instance logging_write_log logging_delete_log "
        "logging_create_log_bucket logging_create_log_sink logging_delete_log_sink",
    ),
    "canvas": (
        "canvas_health_check canvas_get_current_user canvas_list_users canvas_get_user_profile "
        "canvas_list_courses canvas_get_course canvas_list_assignments canvas_get_assignment "
        "canvas_get_submission canvas_list_files canvas_get_file canvas_list_folders canvas_list_pages "
        "canvas_get_page canvas_list_calendar_events canvas_get_upcoming_assignments canvas_get_dashboard "
        "canvas_get_dashboard_cards canvas_get_course_grades canvas_get_user_grades canvas_list_modules "
        "canvas_get_module canvas_list_module_items canvas_get_module_item canvas_list_discussion_topics "
        "canvas_get_discussion_topic canvas_list_announcements canvas_list_quizzes canvas_get_quiz "
        "canvas_get_quiz_questions canvas_list_rubrics canvas_get_rubric canvas_list_conversations "
        "canvas_get_conversation canvas_list_notifications canvas_get_account canvas_list_account_courses "
        "canvas_list_account_users canvas_list_sub_accounts canvas_get_account_reports canvas_get_syllabus",
        "canvas_login canvas_logout canvas_update_user_profile canvas_create_course canvas_update_course "
        "canvas_create_assignment canvas_update_assignment canvas_submit_assignment canvas_submit_grade "
        "canvas_enroll_user canvas_mark_module_item_complete canvas_create_module canvas_create_module_item "
        "canvas_post_to_discussion canvas_create_quiz canvas_start_quiz_attempt canvas_update_quiz "
        "canvas_publish_quiz canvas_delete_quiz canvas_add_quiz_question canvas_update_quiz_question "
        "canvas_delete_quiz_question canvas_create_rubric canvas_create_conversation canvas_create_user "
        "canvas_create_account_report",
    ),
    "woocommerce": (
        "woo_products_list woo_products_get woo_products_variations_list woo_products_categories_list "
        "woo_products_tags_list woo_products_reviews_list woo_orders_list woo_orders_get woo_customers_list "
        "woo_customers_get woo_coupons_list woo_coupons_get woo_reports_sales woo_reports_top_sellers "
        "woo_reports_customers woo_reports_orders woo_reports_products woo_reports_stock woo_reports_low_stock "
        "woo_shipping_zones_list woo_shipping_zones_get woo_shipping_zone_methods_list woo_tax_rates_list "
        "woo_tax_rates_get woo_tax_classes_list woo_system_status woo_system_tools_list woo_settings_list "
        "woo_settings_get woo_payment_gateways_list woo_payment_gateways_get woo_webhooks_list",
        "woo_products_create woo_products_update woo_products_delete woo_products_batch_update "
        "woo_products_categories_create woo_orders_create woo_orders_update woo_orders_delete "
        "woo_orders_batch_update woo_orders_notes_create woo_orders_refunds_create woo_customers_create "
        "woo_customers_update woo_coupons_create woo_coupons_update woo_coupons_delete "
        "woo_shipping_zones_create woo_shipping_zones_update woo_shipping_zone_methods_create "
        "woo_tax_rates_create woo_system_tools_run woo_payment_gateways_update woo_webhooks_create",
    ),
    "filesystem": (
        "read_file read_text_file read_media_file read_multiple_files list_directory list_directory_with_sizes "
        "directory_tree search_files get_file_info list_allowed_directories",
        "write_file append_file edit_file create_directory move_file copy_file",
    ),
    "memory": (
        "read_graph search_nodes open_nodes",
        "create_entities create_relations add_observations delete_entities delete_observations delete_relations",
    ),
    "excel": (
        "validate_formula_syntax read_data_from_excel get_workbook_metadata get_merged_cells "
        "validate_excel_range get_data_validation_info",
        "apply_formula format_range write_data_to_excel create_workbook create_worksheet create_chart "
        "create_pivot_table create_table copy_worksheet delete_worksheet rename_worksheet merge_cells "
        "unmerge_cells copy_range delete_range insert_rows insert_columns delete_sheet_rows delete_sheet_columns",
    ),
    "pdf_tools": (
        "read_pdf_pages get_pdf_info search_pdf_content search_pdf_next_page search_pdf_prev_page "
        "search_pdf_go_page search_pdf_info",
        "merge_pdfs extract_pdf_pages",
    ),
    "terminal": ("show_security_rules", "run_command"),
    "python_execute": ("", "python_execute"),
    "claim_done": ("", "claim_done"),
}


def classify_loca_tool(name: str, server_types: Mapping[str, str]) -> str:
    """Resolve actual registered names, including Canvas's double prefix."""
    matches: set[tuple[str, str]] = set()
    for server, kind in server_types.items():
        if kind not in _CATALOG:
            raise ValueError(f"LOCA capability catalog has no MCP service {kind!r}.")
        for mode, names in zip(("I", "E"), _CATALOG[kind]):
            for raw in names.split():
                if name in {raw, f"{server}_{raw}"}:
                    matches.add((server, mode))
    if len(matches) != 1:
        raise ValueError(f"LOCA tool needs explicit capability review: {name!r}; matches={sorted(matches)}")
    return next(iter(matches))[1]
