import kagglehub

# Download latest version
path = kagglehub.dataset_download("nexuswho/fabric-defects-dataset")

print("Path to dataset files:", path)