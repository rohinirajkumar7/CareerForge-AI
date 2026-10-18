import pytest

from privacy.redactor import guess_name, redact


@pytest.mark.parametrize(
    "text,placeholder",
    [
        ("mail me at jane.doe@example.com today", "[EMAIL]"),
        ("call +91 98765 43210 now", "[PHONE]"),
        ("call (080) 4123 9876", "[PHONE]"),
        ("call 9876543210", "[PHONE]"),
        ("see linkedin.com/in/jane-doe", "[LINKEDIN]"),
        ("code at https://github.com/janedoe/repo", "[GITHUB]"),
        ("portfolio https://janedoe.example.com/work", "[URL]"),
        ("PAN ABCDE1234F", "[ID_NUMBER]"),
        ("Aadhaar 2345 6789 0123", "[ID_NUMBER]"),
        ("SSN 123-45-6789", "[ID_NUMBER]"),
        ("DOB: 12/08/2002", "[DOB]"),
        ("Address: 14 MG Road, Bengaluru", "[ADDRESS]"),
    ],
)
def test_each_pii_type_is_replaced(text, placeholder):
    assert placeholder in redact(text, detect_name=False).text


@pytest.mark.parametrize(
    "text",
    [
        "Worked from 2021-2023 on a team of 12.",
        "Upgraded to Python 3.11.4 and Django 4.2.7.",
        "Uptime of 99.95% across 3 regions.",
        "Server 192.168.100.100 behind Nginx.",
        "Processed 10000 records per second.",
        "Scored 8.9 CGPA and 94.6 percent (2019).",
    ],
)
def test_non_personal_numbers_are_left_alone(text):
    assert redact(text, detect_name=False).text == text


def test_name_is_found_from_first_line_and_removed_everywhere():
    result = redact("Priya Sharma\nSkills: Python\nPriya built a pipeline.")
    assert "Priya" not in result.text and "Sharma" not in result.text
    assert result.counts["NAME"] == 2  # full name once, first name once


def test_explicit_names_and_name_detection_switch():
    assert "Rao" not in redact("Contact: A. Rao", names=["Rao"], detect_name=False).text
    assert redact("Python Developer\nSkills", detect_name=True).counts.get("NAME") is None


@pytest.mark.parametrize("line", ["Curriculum Vitae", "Software Engineer", "Data Analyst Intern", "Page 1 of 2"])
def test_headings_are_not_mistaken_for_names(line):
    assert guess_name(f"{line}\nSkills") is None


def test_redaction_is_idempotent_and_counts_total():
    once = redact("Jane Doe\njane@example.com\n+1 415-555-0132")
    twice = redact(once.text, detect_name=False)
    assert twice.text == once.text and once.total >= 3


def test_labels_are_kept_around_redacted_values():
    out = redact("DOB: 12/08/2002\nAddress: 14 MG Road", detect_name=False).text
    assert out == "DOB: [DOB]\nAddress: [ADDRESS]"
