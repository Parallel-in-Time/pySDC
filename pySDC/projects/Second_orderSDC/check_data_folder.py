import os

folder_name = "./data"

# Check if the folder already exists
if not os.path.isdir(folder_name):  # pragma: no cover
    # Create the folder
    os.makedirs(folder_name)
else:
    pass
