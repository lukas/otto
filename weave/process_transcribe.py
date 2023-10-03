import re
import sys


def strip_ansi_codes(s):
    return re.sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?', '', s)


def process_transcribe(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [strip_ansi_codes(line) for line in lines]
        lines = [line.strip() for line in lines]
        for line in lines:
            if (line == "" or line.startswith("(") or line.startswith("[") or line.startswith("*") or line.startswith(".") or line.startswith("-")):
                pass
            else:
                print(line)


if (__name__ == "__main__"):
    process_transcribe(sys.argv[1])
