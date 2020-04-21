import argparse

def add(a, b):
    return a + b

parser = argparse.ArgumentParser()
parser.set_defaults(func=add)

args = parser.parse_args()

print(args)
print(args.func(1, 2))