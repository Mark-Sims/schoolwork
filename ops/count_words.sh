#! /bin/bash

echo "count these words count these count" | tr [:space:] '\n' | grep -v "^\s*$" | sort | uniq -c | sort -bnr

# tr just replaces spaces with newlines
# grep -v "^\s*$" trims out empty lines
# sort to prepare as input for uniq
# uniq -c to count occurrences
# sort -bnr sorts in numeric reverse order while ignoring whitespace
