import kagglehub

# Download latest version
path = kagglehub.dataset_download("nikhilshingadiya/tinyimagenet200")

print("Path to dataset files:", path)