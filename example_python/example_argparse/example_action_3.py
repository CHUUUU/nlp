import argparse

class plus_1(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, int(values)+1)  # argparse 에 있는 기본 namespace 에 dest 의 key 에 value 를 저장


parser = argparse.ArgumentParser()
parser.add_argument('integers', metavar='N', type=int, nargs='+')
parser.add_argument('-p', '--plus', action=plus_1, dest="run_plus_1", default=2)

args = parser.parse_args()
print(args)
print(args.run_plus_1)

# python .\example_action_3.py 1 2 3 -p 10
# Namespace(integers=[1, 2, 3], run_plus_1=11)
# 11


# python .\example_action_3.py 1 2 3
# Namespace(integers=[1, 2, 3], run_plus_1=2)  # -p 에 값을 안주었으므로, action 은 발동하지 않고 default 를 return
# 2

