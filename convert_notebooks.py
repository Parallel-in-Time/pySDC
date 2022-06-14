import os
'''
Convert notebooks
'''
notebooks = []

# add your notebook here
notebooks.append({'source': 'pySDC/projects/Resilience/Adaptivity.ipynb', 'dest': 'docs/source/projects/Resilience'})
notebooks.append({'source': 'pySDC/projects/Resilience/HotRod.ipynb', 'dest': 'docs/source/projects/Resilience'})

convert_options = '--to rst --execute'

for i in range(len(notebooks)):
    # lint the notebooks
    ret = os.system(f"flake8_nb {notebooks[i]['dest']}")

    # convert them to rst
    if 'dest' in notebooks[i].keys():
        command = f"jupyter-nbconvert --output-dir={notebooks[i]['dest']} {notebooks[i]['source']} {convert_options}"
    else:
        command = f"jupyter-nbconvert {notebooks[i]['source']} {convert_options}"
    os.system(command)
