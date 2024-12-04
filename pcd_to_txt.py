# Define the file paths
pcd_file_path = '/home/chihsun/shared_dir/1204/autoware-241204.pcd'  # Replace with your PCD file path
output_txt_file = '/home/chihsun/Downloads/autoware-241204.txt'  # Replace with desired TXT output path

# Parse the PCD file and extract the point cloud data
header_ended = False
point_data_2d = []

with open(pcd_file_path, 'r') as file:
    for line in file:
        if header_ended:
            # Extract x, y (omit z and other fields if present)
            parts = line.strip().split()
            if len(parts) >= 3:  # Ensure the line contains enough data
                x, y = parts[0], parts[1]  # Take only x and y
                point_data_2d.append(f"{x} {y}")
        elif line.strip() == "DATA ascii":
            header_ended = True

# Write the 2D point data to a TXT file
with open(output_txt_file, 'w') as file:
    file.write("\n".join(point_data_2d))

print(f"Converted 3D PCD to 2D TXT and saved as {output_txt_file}")
