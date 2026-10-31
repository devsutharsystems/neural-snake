"""README.md numbers must match the committed result files (scripts/check_readme_numbers.py), and the check must be able to fail."""
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import check_readme_numbers as crn  # noqa: E402

README = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()


def test_readme_matches_result_files():
    assert crn.check(README) == []


@pytest.mark.parametrize("old,new", [
    ("0.9659 | 0.9467", "0.9660 | 0.9467"),                    # a Kaplan-Meier cell
    ("| 19 | 170 | 11 |", "| 18 | 170 | 11 |"),                 # a rollout count in a table
    ("19 mismatched", "18 mismatched"),                          # a prose count
    ("99.987%", "99.986%"),                                      # the headline accuracy
    ("| 3 | webgpu first | WebGPU | 4.00 |", "| 3 | webgpu first | WebGPU | 4.10 |"),  # a benchmark cell
    ("4 of 1,896", "5 of 1,896"),                                # missed game-overs
    ("`sampleCell()`", "`sampleCel()`"),                         # a function missing from the inventory
    ("1,503 s", "1,500 s"),                                      # training time
    ("Number of nodes: 74", "Number of nodes: 75"),              # browser fact
])
def test_tampered_readme_is_caught(old, new):
    assert old in README, f"test setup: '{old}' not found in README"
    assert crn.check(README.replace(old, new)) != []  # every occurrence changed: a number quoted twice must not hide a mismatch
