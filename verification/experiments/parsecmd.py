#!/usr/bin/env python
import sys

if not sys.stdin.isatty():
    input_stream = sys.stdin
else:
    filename = sys.argv[1]
    input_stream = open(filename,"r")

lines = [line for line in input_stream]

if len(lines) > 0:
    gen = 0
    solve = 0
    for l in lines:
        if("Query Generated in" in l):
            num = l.split(" ")[-1]
            while(num[-1] == "\n" or num[-1] =="s"):
                num = num[0:-1]
            gen = gen + float(num)
        if("Proved in" in l):
            num = l.split(" ")[-1]
            while(num[-1] == "\n" or num[-1] =="s"):
                num = num[0:-1]
            solve = solve + float(num)

    print(gen)
    print(solve)

