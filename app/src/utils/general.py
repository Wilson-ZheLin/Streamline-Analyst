from collections import Counter
import re
import numpy as np
import pandas as pd


def detect_decimal_separator(number_string):
    # Remove all non-digit characters except for commas and periods
    cleaned_number_string = re.sub(r"[^\d,\.]", "", number_string)

    # Split the string based on comma and period
    comma_parts = cleaned_number_string.split(",")
    period_parts = cleaned_number_string.split(".")

    # Check if parts after split contain exactly one segment with less than 3 digits, identifying it as a possible decimal separator
    comma_decimal = any(len(part) == 3 for part in comma_parts[:-1]) and len(comma_parts[-1]) != 3
    period_decimal = any(len(part) == 3 for part in period_parts[:-1]) and len(period_parts[-1]) != 3

    # Detect based on the rules given
    if comma_decimal and not period_decimal:
        return "."
    elif period_decimal and not comma_decimal:
        return ","
    else:
        return None  # Unable to determine based on given heuristics


def clean_value(value):
    """
    Remove all alphabetic characters and special characters except for '.' and ',' from the value.

    :param value: The string value to be cleaned.
    :return: The cleaned value.
    """
    if pd.isna(value):
        return value
    # Remove all characters except digits, '.' and ','
    cleaned_value = re.sub(r"[^0-9.,]", "", value)
    return cleaned_value


def parse_float(value, decimal_separator=None):
    """
    Parse a string to a float, handling various cases such as commas and different decimal separators.

    :param value: The string value to be parsed.
    :return: The parsed float value, or NaN if parsing fails.
    """
    try:
        if decimal_separator:
            value = value.replace(decimal_separator, "")
        if "," in value:
            value = value.replace(",", ".")
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def find_most_frequent(lst):
    """
    Find the value that appears most frequently in the list.

    :param lst: The list to be analyzed.
    :return: A tuple containing the most frequent value and its count.
    """
    if not lst:
        return None, 0
    # Count the occurrences of each element in the list
    counter = Counter(lst)
    # Find the most common element
    most_common_element, count = counter.most_common(1)[0]
    return most_common_element, count


def find_decimal_separator(df, columns_to_check=[]):
    """
    Find the most common decimal separator in the specified columns of the DataFrame.

    :param df: Pandas DataFrame to be processed.
    :param columns_to_check: List of column names to be checked for decimal separators.
    :return: The most common decimal separator found in the specified columns.
    """
    separators = []
    for column in columns_to_check:
        # Detect decimal separators in the column
        detected_separators = [detect_decimal_separator(value) for value in df[column] if not pd.isna(value)]
        # Find the most common decimal separator
        separator, _ = find_most_frequent(detected_separators)
        separators.append(separator)
    # Find the most common decimal separator among all columns
    most_common_separator, _ = find_most_frequent(separators)
    return most_common_separator
