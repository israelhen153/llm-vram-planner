#!/usr/bin/env python3
"""The cold check of chore/sabotage-readme-dedupe: a second row for one driver in
this directory's README table, written so the first version of the row test missed it.

That test matched one exact spacing, `| \\`name\\` |`. A row the README renders the
same way passed it in five other shapes, and a row for a driver that does not exist
passed without backticks. These keep each shape, plus three that the fix closes as a
class:
- a row with no edge pipes;
- a linked name with no extension;
- the table cut in two by a blank line, which leaves every row below it outside the
  table.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

DOC = "tests/sabotage/README.md"
R6 = "| `engine_r6_amd_rows.py` | 6 |"
W1 = "| `workflow_r1_gate.py` | 1 |"


def row_before(anchor, row):
    return [(DOC, anchor, row + "\n" + anchor, 1)]


S = {
 'D1 a second row with its own text':                  row_before(R6, "| `engine_r6_amd_rows.py` | 6 | The AMD rows, again |"),
 'D2 two spaces after the name cell':                  row_before(R6, "| `engine_r6_amd_rows.py`  | 6 | other |"),
 'D3 the name without backticks':                      row_before(R6, "| engine_r6_amd_rows.py | 6 | other |"),
 'D4 a leading space before the row':                  row_before(R6, " | `engine_r6_amd_rows.py` | 6 | other |"),
 'D5 no padding inside the pipes':                     row_before(R6, "|`engine_r6_amd_rows.py`|6|other|"),
 'D6 the name in bold':                                row_before(R6, "| **`engine_r6_amd_rows.py`** | 6 | other |"),
 'D7 no edge pipes, which GitHub still renders as a row': row_before(R6, "`engine_r6_amd_rows.py` | 6 | other"),
 'D8 a linked name with no extension':                 row_before(R6, "| [engine_r6_amd_rows](engine_r6_amd_rows.py) | 6 | other |"),
 'P1 a row for a driver that does not exist, no backticks': row_before(R6, "| engine_r99_not_a_driver.py | 99 | other |"),
 'T1 a blank line cuts the table, and the rows below it leave it': [(DOC, W1, "\n" + W1, 1)],
}

if __name__ == "__main__":
    run_driver(S)
