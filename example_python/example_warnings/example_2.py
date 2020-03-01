import warnings
import requests
import sys

'''
python warnings

default = print the first occurrence of matching warnings for each location (module + number line)
error = turn matching warnings into exceptions
ignore = never print matching warnings
always = always print matching warnings
module = print the first occurrence of matching warnings for each module where the warning is issued (regardless of line number)
once = print only the first occurrence of matching warnings, regardless of location
'''

warnings.filterwarnings("once")
print(sys.argv)
print("#"*10)
if len(sys.argv) == 2:
    Program, URL = sys.argv
    if "http://" in URL:
        warnings.warn('the connection is insecure')
    Response = requests.get(url=URL)
    warnings.warn('the connection is insecure')
else:
    warnings.warn("missing URL")
    warnings.warn("missing URL")

# python example.py
# ['example.py']
# ##########
# example.py:25: UserWarning: missing URL
#   warnings.warn("missing URL")


# python example.py http://google.com
# ['example.py', 'http://google.com']
# ##########
# example.py:21: UserWarning: the connection is insecure
#   warnings.warn('the connection is insecure')


# warn 속성이 once 가 되면서 같은거는 다시 print 하지 않는다.