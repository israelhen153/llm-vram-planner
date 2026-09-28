#!/usr/bin/env python3
"""Round 5 against tests/workflow.test.py: the schedule GitHub delays and drops.

The price job was scheduled at "0 6 * * 1", the start of the hour, which GitHub's
docs name as the time scheduled runs are delayed, and dropped when the load is
high enough. Neither Monday it was due ran on time: 2026-09-21's started at 11:49
UTC, and 2026-09-28's had not started by 07:44, when it was run by hand. The fix
moved it to minute 17, and a check now refuses the start of the hour.

A check comparing the minute field with the string "0" would pass most of these.
They put minute 0 back in the shapes cron allows (00, a step, a list, a star),
take the schedule away so the rule has nothing to judge, schedule a DIFFERENT
workflow on the hour so a rule that reads only price-refresh.yml passes, and quote
`on:` so a reader keyed to YAML 1.1's boolean True finds no triggers at all.
"""
import os, sys
sys.dont_write_bytecode = True   # see the note in harness.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import run_driver

WF = ".github/workflows/price-refresh.yml"
TESTS_WF = ".github/workflows/tests.yml"
CRON = '    - cron: "17 6 * * 1"'
CRON_LINE = '    - cron: "17 6 * * 1" # Monday 06:17 UTC\n'


def minute(expr):
    return [(WF, CRON, f'    - cron: "{expr}"', 1)]


S = {
 'H1 back on the hour':                          minute("0 6 * * 1"),
 'H2 on the hour, written 00':                   minute("00 6 * * 1"),
 'H3 every half hour, so :00 among them':        minute("*/30 6 * * 1"),
 'H4 a list that holds minute 0':                minute("0,17 6 * * 1"),
 'H5 every minute of the hour':                  minute("* 6 * * 1"),
 'H6 the schedule emptied, so the job never runs on its own': [(WF, CRON_LINE, "", 1)],
 'H7 another workflow scheduled on the hour':    [
     (TESTS_WF, "\non:\n  pull_request:\n", '\non:\n  schedule:\n    - cron: "0 3 * * *"\n  pull_request:\n', 1)],
 'H8 `on:` quoted, with the price job back on the hour': [
     (WF, "\non:\n  schedule:\n", '\n"on":\n  schedule:\n', 1),
     (WF, CRON, '    - cron: "0 6 * * 1"', 1)],
}

if __name__ == "__main__":
    run_driver(S)
