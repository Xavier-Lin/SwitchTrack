# -*- coding: utf-8 -*-

# Copyright (c) 2012 Georgios Verigakis <verigak@gmail.com>
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from __future__ import unicode_literals

import sys

from functools import partial
from collections import deque
from datetime import timedelta
from math import ceil
from sys import stderr
try:
    from time import monotonic
except ImportError:
    from time import time as monotonic


__version__ = '1.6'

HIDE_CURSOR = '\x1b[?25l'
SHOW_CURSOR = '\x1b[?25h'


class Infinite(object):
    file = stderr
    sma_window = 10         # Simple Moving Average window
    check_tty = True
    hide_cursor = True

    def __init__(self, message='', **kwargs):
        self.index = 0
        self.start_ts = monotonic()
        self.avg = 0
        self._avg_update_ts = self.start_ts
        self._ts = self.start_ts
        self._xput = deque(maxlen=self.sma_window)
        for key, val in kwargs.items():
            setattr(self, key, val)

        self._max_width = 0
        self._hidden_cursor = False
        self.message = message

        if self.file and self.is_tty():
            if self.hide_cursor:
                print(HIDE_CURSOR, end='', file=self.file)
                self._hidden_cursor = True
        self.writeln('')

    def __del__(self):
        if self._hidden_cursor:
            print(SHOW_CURSOR, end='', file=self.file)

    def __getitem__(self, key):
        if key.startswith('_'):
            return None
        return getattr(self, key, None)

    @property
    def elapsed(self):
        return int(monotonic() - self.start_ts)

    @property
    def elapsed_td(self):
        return timedelta(seconds=self.elapsed)

    def update_avg(self, n, dt):
        if n > 0:
            xput_len = len(self._xput)
            self._xput.append(dt / n)
            now = monotonic()
            # update when we're still filling _xput, then after every second
            if (xput_len < self.sma_window or
                    now - self._avg_update_ts > 1):
                self.avg = sum(self._xput) / len(self._xput)
                self._avg_update_ts = now

    def update(self):
        pass

    def start(self):
        pass

    def writeln(self, line):
        if self.file and self.is_tty():
            width = len(line)
            if width < self._max_width:
                # Add padding to cover previous contents
                line += ' ' * (self._max_width - width)
            else:
                self._max_width = width
            print('\r' + line, end='', file=self.file)
            self.file.flush()

    def finish(self):
        if self.file and self.is_tty():
            print(file=self.file)
            if self._hidden_cursor:
                print(SHOW_CURSOR, end='', file=self.file)
                self._hidden_cursor = False

    def is_tty(self):
        try:
            return self.file.isatty() if self.check_tty else True
        except AttributeError:
            msg = "%s has no attribute 'isatty'. Try setting check_tty=False." % self
            raise AttributeError(msg)

    def next(self, n=1):
        now = monotonic()
        dt = now - self._ts
        self.update_avg(n, dt)
        self._ts = now
        self.index = self.index + n
        self.update()

    def iter(self, it):
        self.iter_value = None
        with self:
            for x in it:
                self.iter_value = x
                yield x
                self.next()
        del self.iter_value

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

COLORS = ('black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan',
          'white')
STYLES = ('bold', 'faint', 'italic', 'underline', 'blink', 'blink2',
          'negative', 'concealed', 'crossed')

def color(s, fg=None, bg=None, style=None):
    sgr = []

    if fg:
        if fg in COLORS:
            sgr.append(str(30 + COLORS.index(fg)))
        elif isinstance(fg, int) and 0 <= fg <= 255:
            sgr.append('38;5;%d' % int(fg))
        else:
            raise Exception('Invalid color "%s"' % fg)

    if bg:
        if bg in COLORS:
            sgr.append(str(40 + COLORS.index(bg)))
        elif isinstance(bg, int) and 0 <= bg <= 255:
            sgr.append('48;5;%d' % bg)
        else:
            raise Exception('Invalid color "%s"' % bg)

    if style:
        for st in style.split('+'):
            if st in STYLES:
                sgr.append(str(1 + STYLES.index(st)))
            else:
                raise Exception('Invalid style "%s"' % st)

    if sgr:
        prefix = '\x1b[' + ';'.join(sgr) + 'm'
        suffix = '\x1b[0m'
        return prefix + s + suffix
    else:
        return s


# Foreground shortcuts
black = partial(color, fg='black')
red = partial(color, fg='red')
green = partial(color, fg='green')
yellow = partial(color, fg='yellow')
blue = partial(color, fg='blue')
magenta = partial(color, fg='magenta')
cyan = partial(color, fg='cyan')
white = partial(color, fg='white')

# Style shortcuts
bold = partial(color, style='bold')
faint = partial(color, style='faint')
italic = partial(color, style='italic')
underline = partial(color, style='underline')
blink = partial(color, style='blink')
blink2 = partial(color, style='blink2')
negative = partial(color, style='negative')
concealed = partial(color, style='concealed')
crossed = partial(color, style='crossed')

class Progress(Infinite):
    def __init__(self, *args, **kwargs):
        super(Progress, self).__init__(*args, **kwargs)
        self.max = kwargs.get('max', 100)

    @property
    def eta(self):
        return int(ceil(self.avg * self.remaining))

    @property
    def eta_td(self):
        return timedelta(seconds=self.eta)

    @property
    def percent(self):
        return self.progress * 100

    @property
    def progress(self):
        if self.max == 0:
            return 0
        return min(1, self.index / self.max)

    @property
    def remaining(self):
        return max(self.max - self.index, 0)

    def start(self):
        self.update()

    def goto(self, index):
        incr = index - self.index
        self.next(incr)

    def iter(self, it):
        try:
            self.max = len(it)
        except TypeError:
            pass

        self.iter_value = None
        with self:
            for x in it:
                self.iter_value = x
                yield x
                self.next()
        del self.iter_value

class Bar(Progress):
    width = 32
    suffix = '%(index)d/%(max)d'
    bar_prefix = ' |'
    bar_suffix = '| '
    empty_fill = ' '
    fill = '#'
    color = None

    def update(self):
        filled_length = int(self.width * self.progress)
        empty_length = self.width - filled_length

        message = self.message % self
        bar = color(self.fill * filled_length, fg=self.color)
        empty = self.empty_fill * empty_length
        suffix = self.suffix % self
        line = ''.join([message, self.bar_prefix, bar, empty, self.bar_suffix,
                        suffix])
        self.writeln(line)


class ChargingBar(Bar):
    suffix = '%(percent)d%%'
    bar_prefix = ' '
    bar_suffix = ' '
    empty_fill = '∙'
    fill = '█'


class FillingSquaresBar(ChargingBar):
    empty_fill = '▢'
    fill = '▣'


class FillingCirclesBar(ChargingBar):
    empty_fill = '◯'
    fill = '◉'


class IncrementalBar(Bar):
    if sys.platform.startswith('win'):
        phases = (u' ', u'▌', u'█')
    else:
        phases = (' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█')

    def update(self):
        nphases = len(self.phases)
        filled_len = self.width * self.progress
        nfull = int(filled_len)                      # Number of full chars
        phase = int((filled_len - nfull) * nphases)  # Phase of last char
        nempty = self.width - nfull                  # Number of empty chars

        message = self.message % self
        bar = color(self.phases[-1] * nfull, fg=self.color)
        current = self.phases[phase] if phase > 0 else ''
        empty = self.empty_fill * max(0, nempty - len(current))
        suffix = self.suffix % self
        line = ''.join([message, self.bar_prefix, bar, current, empty,
                        self.bar_suffix, suffix])
        self.writeln(line)


class PixelBar(IncrementalBar):
    phases = ('⡀', '⡄', '⡆', '⡇', '⣇', '⣧', '⣷', '⣿')


class ShadyBar(IncrementalBar):
    phases = (' ', '░', '▒', '▓', '█')
