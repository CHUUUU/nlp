import argparse
parser = argparse.ArgumentParser()
subparser = parser.add_subparsers()

parser_a = subparser.add_parser("a")
parser_a.add_argument("-x")
print(parser.parse_args(["a", "-x", "1"]))

parser_b = subparser.add_parser("b")
parser_b.add_argument("-y")
print(parser.parse_args(["b", "-y", "2"]))

# Namespace(x='1')
# Namespace(y='2')

