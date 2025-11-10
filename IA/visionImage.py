import kagglehub
import preprocess

# # Download latest version
# path = kagglehub.dataset_download("nexuswho/fabric-defects-dataset")

# print("Path to dataset files:", path)

image = "TexVision/dataset/Fabric Defect Dataset/defect free/809af08c6a3824711208578173.jpg"

image = preprocess.preprocess_image(image)

print(image)