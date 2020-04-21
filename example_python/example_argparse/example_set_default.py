import argparse

parser = argparse.ArgumentParser()
parser.add_argument('foo', type=int)
parser.set_defaults(bar=42, baz='badger')

print(parser.parse_args())


# python .\example_set_default.py 777
# Namespace(bar=42, baz='badger', foo=777)