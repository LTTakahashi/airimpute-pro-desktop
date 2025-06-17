#!/bin/bash
# pkg-config wrapper to filter out libsoup-3.0 references
# This prevents our app from being directly linked against libsoup3

# Get the original output from the real pkg-config
original_flags=$(/usr/bin/pkg-config "$@")

# Filter out any flags related to libsoup-3.0
# This regex is robust enough to catch -lsoup-3.0, -I/path/to/soup-3.0, etc.
filtered_flags=$(echo "$original_flags" | sed -E 's/(-[L|l]|-I)[^ ]*soup-3\.0[^ ]*//g' | sed 's/  */ /g')

# Also replace any remaining -lsoup-3.0 with -lsoup-2.4
filtered_flags=$(echo "$filtered_flags" | sed 's/-lsoup-3\.0/-lsoup-2.4/g')

# Echo the cleaned flags
echo "$filtered_flags"