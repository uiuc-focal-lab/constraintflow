#!/usr/bin/env python
import sys

if not sys.stdin.isatty():
    input_stream = sys.stdin
else:
    filename = sys.argv[1]
    input_stream = open(filename,"r")

lines = [line for line in input_stream]

if len(lines) > 0:
    start = float(lines[0].split(" ")[1])
    gen = 0
    solve = 0
    for l in lines:
        if(l.split(" ")[0] == "gen"):
            gen = gen + float(l.split(" ")[1]) - start
            start = float(l.split(" ")[1])
        if(l.split(" ")[0] == "end"):
            solve = solve + float(l.split(" ")[1]) - start
            start = float(l.split(" ")[1])

    print(gen)
    print(solve)

