import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--foo', action='store')  # default
parser.add_argument('-b', '--bar', action='append')  # array
args = parser.parse_args()

print('args.foo:', args.foo)
print('args.bar:', args.bar)


# python .\example_action.py -f 1 -b 2
# args.foo: 1
# args.bar: ['2']


# python .\example_action.py -f 1 -b 2 -b 3
# args.foo: 1
# args.bar: ['2', '3']
