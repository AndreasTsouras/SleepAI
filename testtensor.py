import re

# Path to your crop file
crop_file = "crop_values.txt"

# Compile a regex pattern that matches lines in the following format:
# "001:bl:250,350, br:550,550, tl:620,75, tr:800,100"
pattern = re.compile(
    r'^(?P<patient>\d+):bl:(?P<bl_x>\d+),(?P<bl_y>\d+),\s*br:(?P<br_x>\d+),(?P<br_y>\d+),\s*tl:(?P<tl_x>\d+),(?P<tl_y>\d+),\s*tr:(?P<tr_x>\d+),(?P<tr_y>\d+)$'
)

print("DEBUG: Reading crop values from file:")
with open(crop_file, "r") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    match = pattern.match(line)
    if match:
        patient = match.group("patient")
        bl_x, bl_y = match.group("bl_x"), match.group("bl_y")
        br_x, br_y = match.group("br_x"), match.group("br_y")
        tl_x, tl_y = match.group("tl_x"), match.group("tl_y")
        tr_x, tr_y = match.group("tr_x"), match.group("tr_y")
        print(f"Line {idx+1}: Patient {patient}")
        print(f"  Bottom Left: ({bl_x}, {bl_y})")
        print(f"  Bottom Right: ({br_x}, {br_y})")
        print(f"  Top Left: ({tl_x}, {tl_y})")
        print(f"  Top Right: ({tr_x}, {tr_y})")
    else:
        print(f"Line {idx+1} did not match the expected format: {line}")
