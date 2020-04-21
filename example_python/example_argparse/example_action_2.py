import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.integers)
print(args.accumulate(args.integers))

# nargs = 스위치나 파라미터가 받을 수 있는 값의 개수를 가리킴
# nargs = "*' -> 0개 이상
# nargs = '+' -> 1개 이상
# nargs = '?' : 0개 또는 1개의 값을 읽어들인다.
#               인자와 값을 모두 적은 경우 해당 값이 저장된다.
#               인자만 적은 경우 const 값이 저장된다.
#               아무것도 적지 않았으면 default 값이 저장된다.
# metavar = usage 메시지를 출력할 때 표시할 메타변수이름을 지정해준다.
# type = 파싱하여 저장할 때 타입을 변경할 수 있다
# const=sum, default=max 명령행에 --sum 가 지정되었을 경우 sum() 함수가 되고, 그렇지 않으면 max() 함수가 실행

print(sum([1, 2]))
print(max([1, 2]))

# metavar 를 에러내서 usage 에서 보기
# python .\example_action_2.py
# usage: example_action_2.py [-h] [--sum] N [N ...]
# example_action_2.py: error: the following arguments are required: N


# sum 함수 사용해보기
# python .\example_action_2.py 1 2 3 --sum
# [1, 2, 3]
# 6
# 3
# 2


# max 함수 사용해보기
# python .\example_action_2.py 1 2 3
# [1, 2, 3]
# 3
# 3
# 2
