Order summaries expose internal status keys to customers. For example, an order with the status `in_progress` is currently shown as `Status: in_progress` instead of a readable customer-facing label.

Fix the order-summary behavior so that status values are displayed using the existing human-readable labels. Preserve the existing order number and owner output.

Do not change the tests.
