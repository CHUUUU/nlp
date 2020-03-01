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

print(sys.argv)
print("#"*10)
if len(sys.argv) == 2:
    Program, URL = sys.argv
    if "http://" in URL:
        warnings.warn('the connection is insecure')
    Response = requests.get(url=URL)

else:
    warnings.warn("missing URL")

# ['example.py']
# ##########
# example.py:25: UserWarning: missing URL
#   warnings.warn("missing URL")
