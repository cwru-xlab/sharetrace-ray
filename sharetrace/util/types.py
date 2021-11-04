from datetime import datetime, timedelta
from typing import Union

from numpy import datetime64, timedelta64

TimeDelta = Union[timedelta, timedelta64]
DateTime = Union[datetime, datetime64]
