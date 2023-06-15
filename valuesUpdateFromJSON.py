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
        lines[i] = f'Average magnitude of motion: {data["average_motion"]:.2f}\n'
    elif 'Total magnitude of motion:' in line:
        lines[i] = f'Total magnitude of motion: {data["total_motion"]:.2f}\n'
    elif 'PSNR for NNMP:' in line:
        lines[i] = f'PSNR for NNMP: {data["PSNR_for_NNMP"]:.2f}\n'
    elif 'PSNR for Full Search:' in line:
        lines[i] = f'PSNR for Full Search: {data["PSNR_for_Full_Search"]:.2f}\n'
    elif '- Frame width:' in line:
        lines[i] = f'- Frame width: {data["width"]}\n'
    elif '- Frame height:' in line:
        lines[i] = f'- Frame height: {data["height"]}\n'
    elif '- Frames per second:' in line:
        lines[i] = f'- Frames per second: {data["fps"]}\n'
    elif '- Total frames:' in line:
        lines[i] = f'- Total frames: {data["numFrames"]}\n'
    elif 'The marginal loss between NNMP and Full Search' in line:
        lines[i] = f'The marginal loss between NNMP and Full Search is {data["PSNR_NNMP_vs_FullSearch"]:.2f}% in Peak-Signal-to-Noise ratio (PSNR).\n'
    elif 'The NNMP utilize the time by:' in line:
        lines[i] = f'The NNMP utilize the time by: {data["time_percent"]:,.2f}% .\n'

# Write the modified markdown back to file
with open('README.md', 'w') as f:
    f.writelines(lines)
