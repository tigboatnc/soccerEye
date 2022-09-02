from pathlib import Path

def util_searching_all_files(directory: Path):   
    file_list = [] 
    for x in directory.iterdir():
        if x.is_file():
            if '.mp4' in str(x):
                file_list.append(x)
        else:
            file_list.append(util_searching_all_files(directory/x))
    return file_list