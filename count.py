# Original Author : https://www.cnblogs.com/laizhenghong2012/p/11348004.html
# Adjusted by Kirigaya, including exposure of API and structure adjustment

import os
import sys
import fire
import prettytable as pt

# collection of suffix of python, cpp and java
CPP_SUFFIX_SET    : set = {'.h', '.hpp', '.hxx', '.c', '.cpp', '.cc', '.cxx'}
PYTHON_SUFFIX_SET : set = {'.py'}
JAVA_SUFFIX_SET   : set = {'.java'}

# glabal 
cpp_lines    : int = 0
python_lines : int = 0
java_lines   : int = 0
total_lines  : int = 0

# ignore unimportant files in order to accelerate process
ignore_suffix = [".pyc", ".jpg", ".png", ".jepg", ".gitignore"]
extra_ignore_dir = [".git", "__pycache__"]

# enumerate all the files recursively
def list_files(path : str, ignore_dir : list = []):
    filenames = os.listdir(path)
    for f in filenames:
        # ignore the ignored folder
        fpath = os.path.join(path, f)
        if os.path.isfile(fpath) and not _end_with_ignore_suffix(fpath):
            count_lines(fpath)
        if os.path.isdir(fpath) and f not in ignore_dir:
            list_files(fpath, ignore_dir=ignore_dir)

def _end_with_ignore_suffix(path : str):
    for suffix in ignore_suffix:
        if path.endswith(suffix):
            return True
    return False

# count the lines if fpath is a file instead of a folder
def count_lines(fpath : str):
    global CPP_SUFFIX_SET, PYTHON_SUFFIX_SET, JAVA_SUFFIX_SET
    global cpp_lines, python_lines, java_lines, total_lines

    # count number of line
    with open(fpath, 'rb') as f:
        count = 0
        last_data = '\n'
        while True:
            data = f.read(0x400000)
            if not data:
                break
            count += data.count(b'\n')
            last_data = data
        if last_data[-1:] != b'\n':
            count += 1

    print("{}\t{}".format(fpath, count))
    # we only count cpp, python and java
    # If you want
    suffix = os.path.splitext(fpath)[-1]
    if suffix in CPP_SUFFIX_SET:
        cpp_lines += count
    elif suffix in PYTHON_SUFFIX_SET:
        python_lines += count
    elif suffix in JAVA_SUFFIX_SET:
        java_lines += count
    else:
        pass

# print with a table style
def print_result():
    tb = pt.PrettyTable()
    tb.field_names = ['CPP', 'PYTHON', 'JAVA', 'TOTAL']
    total_lines = cpp_lines + python_lines + java_lines
    tb.add_row([cpp_lines, python_lines, java_lines, total_lines])
    print(tb)

# entrance
def main(path : str, ignore : list = None):
    try:
        ignore_dir = ignore if len(ignore) else []
    except:
        raise ValueError("Syntx Error in argument --ignore, check README.md for details!")
    
    ignore_dir += extra_ignore_dir

    list_files(path, ignore_dir=ignore_dir)
    print_result()

# use fire to expose the API to command
if __name__ == '__main__':
    fire.Fire(main)