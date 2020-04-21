import argparse

parser = argparse.ArgumentParser()
parser.add_argument('foo', type=int)

print(parser.parse_args(['736']))  # command line 대신 여기에 첫번째 arg 를 넣음
print(parser.parse_args([]))

# python .\example_parse_args.py
# Namespace(foo=736)     # 첫번째 print
# usage: example_set_default.py [-h] foo       # 두번째 print 로 인자를 넣지 않아 에러
# example_set_default.py: error: the following arguments are required: foo
