"""各用途的导出器。"""

from .bert import export as export_bert
from .fasttext import export as export_fasttext
from .rf import export as export_rf

EXPORTERS = {
    "bert": export_bert,
    "fasttext": export_fasttext,
    "rf": export_rf,
}
