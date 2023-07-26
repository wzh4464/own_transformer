"""
This is a python script that remove lines including s1 to line including s2 in a file.

Usage:
    python remove_inherit.py <file> <s1> <s2> -o <output_file>
    
    <file> is the file to be processed
    s1 : string
    s2 : string
    <output_file> is the file to be written to
"""

import re
import sys

def remove_inherit(file="doc.md", s1="inherit", s2="-------------", output_file="doc.md"):
    # open the file to be processed
    with open(file, 'r') as f:
        file_content = f.read()
    start = -1
    end = -1
    pair = []
    linenum = -1
    for line in file_content.splitlines():
        linenum += 1
        if start < end and start != -1 and end != -1:
            pair.append((start, end))
            start = -1
            end = -1
        if s1 in line:
            start = linenum
            continue
        # if s2 in line or line has no other characters other than spaces or line is null
        if s2 in line or re.match(r'^\s*$', line):
            end = linenum
            continue
    
    # remove the lines
    while len(pair) > 0:
        to_remove = pair.pop()
        file_content = '\n'.join(file_content.splitlines()[:to_remove[0]] + file_content.splitlines()[to_remove[1]+1:])
    
    # write the file
    with open(output_file, 'w') as f:
        f.write(file_content)
    
    print("Done!")

if __name__ == '__main__':
    # get the arguments
    if len(sys.argv) != 6:
        print("Using default arguments")
        remove_inherit()
    else:
        file = sys.argv[1]
        s1 = sys.argv[2]
        s2 = sys.argv[3]
        output_file = sys.argv[5]
        remove_inherit(file, s1, s2, output_file)