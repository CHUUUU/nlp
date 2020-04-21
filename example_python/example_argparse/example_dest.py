import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--foo', dest='foo_list')
parser.add_argument('-b', '--bar', dest='bar_value')
args = parser.parse_args()

print('args.foo_list:', args.foo_list)
print('args.bar_value:', args.bar_value)

# python .\example_dest.py -f 1 -b 2
# args.foo_list: 1
# args.bar_value: 2

# dest 는 augment 위치를 변경한다.