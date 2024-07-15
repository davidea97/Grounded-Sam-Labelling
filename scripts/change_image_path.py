# Define the path to your text file
txt_file_path = 'file_list.txt'
new_part = '/media/data/Datasets/imagenet21k_resized'  # Define the additional part to add to the path

# Read the file and modify the paths
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# List to store the modified lines
modified_lines = []

# Process each line
for line in lines:
    parts = line.strip().split()
    if len(parts) >= 4:
        # Modify the path by adding the additional part
        path_parts = parts[0].split('/')
        new_path = '/'.join([new_part] + path_parts[:-1] + path_parts[-1:])
        # Reconstruct the line with the modified path
        modified_line = f"{new_path} {parts[1]} {parts[2]} {parts[3]}"
        modified_lines.append(modified_line)

# Write the modified lines back to the file
with open("mod" + txt_file_path, 'w') as file:
    for modified_line in modified_lines:
        file.write(modified_line + '\n')

print(f"File '{txt_file_path}' has been updated with the new paths.")
