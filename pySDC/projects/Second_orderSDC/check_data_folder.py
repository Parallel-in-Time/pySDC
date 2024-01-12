import os  # pragma: no cover

folder_name = "./data"  # pragma: no cover

# Check if the folder already exists
if not os.path.isdir(folder_name):  # pragma: no cover
    # Create the folder
    os.makedirs(folder_name)
else:
    pass
