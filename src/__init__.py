# from .model import DiabetesRegressorModel
# from .data_utils import get_data_splits
# from .config import settings
#
# .
# __all__ = [
#     "DiabetesRegressorModel",
#     "get_data_splits",
#     "settings"
# ]

from .model import DiabetesRegressorModel
from .config import settings
from .data_utils import get_data_splits

# Defining __all__ tells Python what to export when someone runs
# 'from src import *' (though explicit imports are preferred in prod)
__all__ = ["DiabetesRegressorModel", "settings", "get_data_splits"]