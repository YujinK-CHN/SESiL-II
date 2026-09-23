"""
Run logging.

Records go to JSONL -- one JSON object per line, append-only.

The previous approach wrote everything through utils.write_to_csv, which takes
the column header from the FIRST row it ever sees and then appends values
positionally. That is fine for one uniform record type and silently corrupting
for several: a population row, an offspring row and a mutation row carry
different keys, so every row after the first lands under the wrong headers.

JSONL removes the problem rather than working around it -- each record carries
its own keys, streams are separated by purpose, and a truncated file from a
killed run still parses up to the last complete line.

Two streams:

    eval.jsonl   the plotting surface. One record per evaluation watermark,
                 with an identical schema for every method, so SESiL and the
                 baseline can be read into the same frame and plotted on the
                 same axes.
    train.jsonl  everything internal -- certificates, pairings, merges,
                 mutations. Heterogeneous by design; for diagnosis, not for
                 the headline figure.
"""

import json
import os


class JsonlWriter:
    """Append-only JSONL sink."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._handle = None

    def _open(self):
        if self._handle is None:
            self._handle = open(self.path, 'a', encoding='utf-8')
        return self._handle

    def write(self, record):
        """Append one record, flushed immediately so a killed run keeps its log."""
        handle = self._open()
        handle.write(json.dumps(record, default=_coerce) + '\n')
        handle.flush()

    def close(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _coerce(value):
    """Make numpy scalars, arrays and sets JSON-serialisable."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    return str(value)


class RunLogger:
    """The two streams a run writes, plus the console line that mirrors them."""

    def __init__(self, run_dir, quiet=False):
        self.run_dir = run_dir
        self.quiet = quiet
        self.eval = JsonlWriter(os.path.join(run_dir, 'eval.jsonl'))
        self.train = JsonlWriter(os.path.join(run_dir, 'train.jsonl'))

    def log_eval(self, record):
        """One aligned evaluation point. Same schema for every method."""
        self.eval.write(record)
        if not self.quiet:
            print(f'[eval] budget={record.get("budget", 0):.2f}  '
                  f'best={record.get("best_agent_overall", 0):.4f}  '
                  f'oracle={record.get("oracle_overall", 0):.4f}  '
                  f'mean={record.get("population_mean", 0):.4f}')

    def log_train(self, record):
        """One internal event. Schema varies with `stage`."""
        self.train.write(record)

    def close(self):
        self.eval.close()
        self.train.close()
