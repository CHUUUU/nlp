import argparse

def add(a, b):
    return a + b

parser = argparse.ArgumentParser()
parser.set_defaults(func=add)

args = parser.parse_args()

print(args)
print(args.func(1, 2))

# python .\example_set_default_2.py
# Namespace(func=<function add at 0x00000202EEE108B8>)
# 3
