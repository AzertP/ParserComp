#!/usr/bin/env python3
"""
Replace the block between --begin and --end markers (inclusive) in a file
with the contents of a replacement file.

Usage:
    python3 replace.py --begin MARKER_START --end MARKER_END source.txt target.tex
"""

import argparse
import sys


def find_block(lines, begin_marker, end_marker, target_path):
    begin_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if begin_marker in line and begin_idx is None:
            begin_idx = i
        if end_marker in line and begin_idx is not None and end_idx is None:
            end_idx = i
    if begin_idx is None:
        print(f"Error: begin marker {begin_marker!r} not found in {target_path}", file=sys.stderr)
        sys.exit(1)
    if end_idx is None:
        print(f"Error: end marker {end_marker!r} not found after begin marker in {target_path}", file=sys.stderr)
        sys.exit(1)
    return begin_idx, end_idx


def view_block(begin_marker, end_marker, target_path):
    with open(target_path, 'r') as f:
        lines = f.readlines()
    begin_idx, end_idx = find_block(lines, begin_marker, end_marker, target_path)
    print(''.join(lines[begin_idx:end_idx + 1]), end='')


def replace_block(begin_marker, end_marker, source_path, target_path):
    with open(source_path, 'r') as f:
        replacement = f.read()

    with open(target_path, 'r') as f:
        lines = f.readlines()

    begin_idx, end_idx = find_block(lines, begin_marker, end_marker, target_path)

    new_lines = lines[:begin_idx] + [replacement] + lines[end_idx + 1:]

    with open(target_path, 'w') as f:
        f.writelines(new_lines)

    print(f"Replaced lines {begin_idx + 1}–{end_idx + 1} in {target_path}")


def main():
    parser = argparse.ArgumentParser(description="View or replace a marked block in a file.")
    parser.add_argument('--begin', required=True, help='Start marker (inclusive)')
    parser.add_argument('--end',   required=True, help='End marker (inclusive)')
    parser.add_argument('source',  nargs='?',     help='File whose contents replace the block (omit to view)')
    parser.add_argument('target',                 help='File in which the block is found')
    args = parser.parse_args()

    if args.source is None:
        view_block(args.begin, args.end, args.target)
    else:
        replace_block(args.begin, args.end, args.source, args.target)


if __name__ == '__main__':
    main()
