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

warnings.filterwarnings("default")
print(sys.argv)
print("#"*10)
if len(sys.argv) == 2:
    Program, URL = sys.argv
    if "http://" in URL:
        warnings.warn('the connection is insecure')
    Response = requests.get(url=URL)
    warnings.warn('the connection is insecure')
    print("error print")
else:
    warnings.warn("missing URL")
    warnings.warn("missing URL")
    print("error print")

# python example.py
# ['example.py']
# ##########
# example.py:27: UserWarning: missing URL
#   warnings.warn("missing URL")
# example.py:28: UserWarning: missing URL
#   warnings.warn("missing URL")
# error print



# python example.py http://google.com
# ['example.py', 'http://google.com']
# ##########
# example.py:22: UserWarning: the connection is insecure
#   warnings.warn('the connection is insecure')
# example.py:24: UserWarning: the connection is insecure
#   warnings.warn('the connection is insecure')
# error print


# warn 이 기존 warn 속성