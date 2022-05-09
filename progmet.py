import datetime
import sys
import time


def progmet(elems, *args, **kwargs):
    return ProgressMeter(*args, **kwargs)(elems)


class ProgressMeter:

    def __init__(self, title=None, interval_num=0, interval_time=0, num_div=0, print_func=None):
        self._interval_num = interval_num
        self._interval_time = interval_time
        self._num_div = num_div
        self._print_func = print_func or default_print_func
        self._title = title

    def __call__(self, elems):
        self._n = len(elems)
        self._iter = iter(elems)
        self._i = 0  # Number of iterations completed.
        self._start = time.time()
        # State at previous checkpoint.
        self._prev_i = None
        self._prev_time = None
        self._prev_progress = None
        return self

    def __iter__(self):
        return self

    def __next__(self):
        self._increment()
        return next(self._iter)  # May raise StopIteration.

    def _progress(self):
        """Returns numerator for fraction over num_div."""
        return self._i * self._num_div // self._n

    def _increment(self):
        # This is called before next(self._iter) therefore self._i is the
        # number of complete iterations of the for loop (initially zero).
        now = time.time()
        if self._i == 0 or self._at_checkpoint(now):
            # Do not print if zero iterations have been completed.
            if self._i > 0:
                self._print_func(self._title, self._i, self._n, now, self._start, self._prev_i, self._prev_time)
            # Update checkpoint state.
            self._prev_i = self._i
            self._prev_time = now
            self._prev_progress = self._progress()

        # Increment the number of complete iterations for the next call.
        self._i += 1

    def _at_checkpoint(self, now):
        if self._i == self._n:
            return True
        if self._interval_num:
            if self._prev_i is None or self._i - self._prev_i >= self._interval_num:
                return True
        if self._interval_time:
            if self._prev_time is None or now - self._prev_time >= self._interval_time:
                return True
        if self._num_div:
            progress = self._progress()
            if self._prev_progress is None or progress - self._prev_progress >= 1:
                return True
        return False


def default_print_func(title, i, n, now, start_time, prev_i, prev_time):
    time_elapsed = now - start_time
    if i == 0:
        return
    mean_period = float(time_elapsed) / i
    inst_period = float(now - prev_time) / (i - prev_i)
    progress_str = (title + ': ') if title else ''
    if n is None:
        progress_str += '{:d}'.format(i)
    else:
        percent = i / n * 100
        time_total = n * mean_period
        progress_str += '{:3.0f}% ({:d}/{:d})'.format(percent, i, n)
    progress_str += '; T={:#.3g} f={:#.3g}; mean T={:#.3g} f={:#.3g}; elapsed {}'.format(
        inst_period, 1. / inst_period,
        mean_period, 1. / mean_period,
        _format_dur(time_elapsed))
    if n is not None and i < n:
        time_rem = time_total - time_elapsed
        progress_str += '; remaining {} of {}'.format(
            _format_dur(time_rem), _format_dur(time_total))
    print(progress_str, file=sys.stderr)
    sys.stderr.flush()


def _format_dur(seconds):
    return str(datetime.timedelta(seconds=round(seconds)))
