import argparse

def add(a, b):
    return a + b

def minus(a, b):
    return a - b


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers()

parser_a = subparser.add_parser("a")
parser_a.add_argument("-x")
parser_a.set_defaults(func=add)
print(parser.parse_args(["a", "-x", "1"]))

parser_b = subparser.add_parser("b")
parser_b.add_argument("-y")
parser_b.set_defaults(func=minus)
print(parser.parse_args(["b", "-y", "2"]))


args = parser.parse_args()
args_a = parser_a.parse_args()
args_b = parser_b.parse_args()
print(args)
print(args_a.func(1, 2))
print(args_b.func(1, 2))


# python .\example_set_default_3.py
# Namespace(func=<function add at 0x0000025E8C680E58>, x='1')
# Namespace(func=<function minus at 0x0000025E8C74BF78>, y='2')
# Namespace()
# 3
# -1