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

warnings.filterwarnings("error")
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
# Traceback (most recent call last):
#   File "example.py", line 27, in <module>
#     warnings.warn("missing URL")
# UserWarning: missing URL



# python example.py http://google.com
# ['example.py', 'http://google.com']
# ##########
# Traceback (most recent call last):
#   File "example.py", line 22, in <module>
#     warnings.warn('the connection is insecure')
# UserWarning: the connection is insecure


# warn 이 exception 으로 변경되어 warn 에서 멈춤