"""
logtools.py - parsing and text-cleaning helpers for HDFS-style log lines.

An HDFS log line looks like:
    081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 terminating
    date   time   pid level component                     message

Two jobs live here:
  1. parse_line()  - split a raw line into timestamp, level, component, message
  2. mask_message() - replace variable parts (block IDs, IPs, numbers, paths)
                      with placeholders, so lines that describe the same event
                      become identical text. This is what lets text clustering
                      group failures by *type* instead of by *instance*.
"""

import re
from datetime import datetime
from typing import Optional, NamedTuple

LINE_RE = re.compile(
    r'^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+'
    r'(?P<level>[A-Z]+)\s+(?P<component>[^:]+):\s*(?P<message>.*)$'
)
BLOCK_RE = re.compile(r'blk_-?\d+')

# Order matters: block IDs and IPs first, then paths, then any leftover numbers.
_MASKS = [
    (BLOCK_RE, '<BLK>'),
    (re.compile(r'/?\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?'), '<IP>'),
    (re.compile(r'(?:/[\w.\-]+)+'), '<PATH>'),
    (re.compile(r'\b\d+\b'), '<NUM>'),
]


class LogLine(NamedTuple):
    timestamp: datetime
    level: str
    component: str
    message: str


def parse_line(raw: str) -> Optional[LogLine]:
    """Return a LogLine, or None if the line is not in HDFS format."""
    m = LINE_RE.match(raw.strip())
    if not m:
        return None
    ts = datetime.strptime(m['date'] + m['time'], '%y%m%d%H%M%S')
    return LogLine(ts, m['level'], m['component'].strip(), m['message'])


def mask_message(message: str) -> str:
    """Replace variable fields with placeholders, e.g. block IDs -> <BLK>."""
    for pattern, token in _MASKS:
        message = pattern.sub(token, message)
    return re.sub(r'\s+', ' ', message).strip()


def block_ids(message: str) -> list:
    """All HDFS block IDs mentioned in a message (a line can mention several)."""
    return BLOCK_RE.findall(message)
