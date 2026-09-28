"""Allow `python -m panguforge`."""
import sys
from .cli.main import main

sys.exit(main())
