
def dict_to_markdown_table(data):
    assert isinstance(data, dict), "Invalid data. Please provide a dictionary."

    for key, value in data.items():
        if isinstance(value, float):
            data[key] = f"{value:.3f}"
        elif isinstance(value, tuple):
            if len(value) == 2:
                data[key] = f"{value[0]:.3f} ({value[1]:.3f})"
            elif len(value) == 3:
                data[key] = f"{value[0]:.3f} ({value[1]:.3f}, {value[2]:.3f})"
    max_key_length = max(len(str(key)) for key in data.keys())
    max_value_length = max(len(str(value)) for value in data.values())

    headers = ["Metric", "Value"]

    header_row = f"| {headers[0].center(max_key_length)} | {headers[1].center(max_value_length)} |"
    separator_row = f"| {'-' * max_key_length} | {'-' * max_value_length} |"

    # Create the data rows with padding
    data_rows = [
        f"| {str(key).ljust(max_key_length)} | {value.center(max_value_length)} |" for key, value in data.items()
    ]

    # Combine all parts to form the markdown table
    markdown_table = "\n".join([header_row, separator_row] + data_rows)
    return markdown_table
