import sys

from .app import main

if __name__ == '__main__':      # the background process (mapper.py) re-imports this module
    sys.exit(main())
