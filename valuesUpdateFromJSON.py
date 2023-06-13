import json

# Read data from JSON file
with open('res.json', 'r') as f:
    data = json.load(f)

# Read the existing markdown file
with open('README.md', 'r') as f:
    lines = f.readlines()

# Find the line to replace for each data item
for i, line in enumerate(lines):
    if 'Average magnitude of motion:' in line:
        lines[i] = f'Average magnitude of motion: {data["average_motion"]}\n'
    elif 'Total magnitude of motion:' in line:
        lines[i] = f'Total magnitude of motion: {data["total_motion"]}\n'
    elif 'PSNR for NNMP:' in line:
        lines[i] = f'PSNR for NNMP: {data["PSNR_for_NNMP"]}\n'
    elif 'PSNR for Full Search:' in line:
        lines[i] = f'PSNR for Full Search: {data["PSNR_for_Full_Search"]}\n'
    elif '- Frame width:' in line:
        lines[i] = f'- Frame width: {data["width"]}\n'
    elif '- Frame height:' in line:
        lines[i] = f'- Frame height: {data["height"]}\n'
    elif '- Frames per second:' in line:
        lines[i] = f'- Frames per second: {data["fps"]}\n'
    elif '- Total frames:' in line:
        lines[i] = f'- Total frames: {data["numFrames"]}\n'
# Write the modified markdown back to file
with open('README.md', 'w') as f:
    f.writelines(lines)
