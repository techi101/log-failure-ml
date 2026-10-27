"""Unit tests for the log parsing and masking helpers. Run: pytest -q"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from logtools import block_ids, mask_message, parse_line

LINE = ('081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block '
        'blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010')


def test_parse_line_fields():
    line = parse_line(LINE)
    assert line.level == 'INFO'
    assert line.component == 'dfs.DataNode$DataXceiver'
    assert line.timestamp.strftime('%Y-%m-%d %H:%M:%S') == '2008-11-09 20:35:18'
    assert line.message.startswith('Receiving block')


def test_parse_line_rejects_other_formats():
    assert parse_line('[FAIL] test_uart_tx: Timeout after 500ms') is None
    assert parse_line('') is None


def test_mask_replaces_block_ids_and_ips():
    masked = mask_message(parse_line(LINE).message)
    assert masked == 'Receiving block <BLK> src: <IP> dest: <IP>'


def test_same_event_different_values_mask_identically():
    a = mask_message('Deleting block blk_123 file /mnt/hadoop/dfs/data/current/blk_123')
    b = mask_message('Deleting block blk_-987 file /mnt/hadoop/dfs/data/current/blk_-987')
    assert a == b


def test_mask_numbers():
    assert mask_message('PacketResponder 1 for block blk_5 terminating') == \
        'PacketResponder <NUM> for block <BLK> terminating'


def test_block_ids_finds_all():
    msg = 'BLOCK* ask 10.0.0.1:50010 to replicate blk_1 and blk_-2'
    assert block_ids(msg) == ['blk_1', 'blk_-2']
