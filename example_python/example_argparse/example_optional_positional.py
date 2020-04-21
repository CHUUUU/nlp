import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--foo', '-f') # optional
parser.add_argument('bar')         # positional
args = parser.parse_args()
print('args.foo:', args.foo)
print('args.bar:', args.bar)


# python .\test.py -f 1 you
# args.foo: 1
# args.bar: you


# -가 붙어 있으면 optional 아니면 positional 인자
# optional 인자는 지정하지 않아도 되고, 그럴 경우 기본값이 저장된다.
# positional 인자는 반드시 값을 정해 주어야 한다.