#!/usr/bin/env python3
#
# BSD 3-Clause License
#
# Copyright (c) 2023-2024 Fred W6BSD
# All rights reserved.
#
#
"""
Draw a Smith chart graph and more from a .s1p file using a NanoVNA or a RigExpert Zoom.

Example:
% RigExpert -t '6-10m-antenna' --s1p-file 6-10.s1p -A
08/27/24 14:31:15 - 203 INFO - /tmp/6-10m-antenna-all.png
"""

import argparse
import logging
import pathlib
import re
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import skrf as rf  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from scipy import signal  # type: ignore[import-untyped]

__author__ = 'Fred W6BSD - https://gist.github.com/0x9900/'

DPI = 100
LINE_COLOR = "yellow"
SWR_COLOR = "cyan"
LIGHTGRAY = "#ababab"
DARKBLUE = "#1d1330"

MY_PARAMS = {
  'axes.edgecolor': '#4b4b4b',
  'axes.facecolor': DARKBLUE,
  'axes.grid': True,
  'axes.grid.which': 'both',
  'axes.labelcolor': LIGHTGRAY,
  'axes.labelsize': 10,
  'axes.linewidth': 1.5,
  'axes.spines.bottom': True,
  'axes.spines.left': True,
  'axes.spines.right': False,
  'axes.spines.top': False,
  'figure.dpi': 100,
  'figure.edgecolor': DARKBLUE,
  'figure.facecolor': DARKBLUE,
  'figure.figsize': (12, 5),
  'font.size': 10,
  'grid.alpha': 0.7,
  'grid.color': LIGHTGRAY,
  'grid.linestyle': 'dashed',
  'grid.linewidth': 0.25,
  'legend.fontsize': 8,
  'lines.linewidth': 1.5,
  'lines.markersize': 5,
  'text.color': 'white',
  'xtick.color': LIGHTGRAY,
  'xtick.labelcolor': LIGHTGRAY,
  'xtick.labelsize': 8,
  'xtick.minor.visible': False,
  'ytick.color': LIGHTGRAY,
  'ytick.labelcolor': LIGHTGRAY,
  'ytick.labelsize': 8,
  'ytick.minor.visible': False
}


BANDS = (
  (14000, 14350, '20m'),
  (7000, 7300, '40m'),
  (10100, 10150, '30m'),
  (3500, 4000, '80m'),
  (21000, 21450, '15m'),
  (18068, 18168, '17m'),
  (28000, 29700, '10m'),
  (50000, 54000, '6m'),
  (24890, 24990, '12m'),
  (1800, 2000, '160m'),
  (144000, 148000, '2m'),
  (5258, 5450, '60m'),
  (420000, 450000, '0.70m'),
  (219000, 225000, '1.25m'),
  (1240000, 1300000, '0.23m'),
  (10000000, 10500000, '0.02m'),
)


LOG_FORMAT = '%(asctime)s - %(lineno)d %(levelname)s - %(message)s'
logging.basicConfig(format=LOG_FORMAT, datefmt='%x %X', level=logging.INFO)


def slugify(text: str) -> str:
  return re.sub(r'[\W_]+', '-', text.lower())


def read_s1p(filename: pathlib.Path, f_range: List[float] | None) -> rf.Network:
  try:
    network = rf.Network(filename)
  except NotImplementedError:
    raise NotImplementedError('Choose the Touchstone [S,RI] format when exporting your data')
  except ValueError:
    raise ValueError(f'{filename} does not look like a valid Touchstone file.')

  # Often the RigExpert export a wrong last value (Remove the last value)
  network = network[0:-1]
  network.frequency.unit = 'MHz'
  if f_range:
    start = np.argmin(np.abs(network.frequency.f - f_range[0]))
    stop = np.argmin(np.abs(network.frequency.f - f_range[1]))
    network = network[start:stop]

  return network


def smith_chart(dut: rf.Network, ax: Axes) -> None:
  ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to square
  rf.plotting.smith(ax=ax, draw_vswr=[2, 3], draw_labels=True, ref_imm=50.0,
                    chart_type='z')
  dut.plot_s_smith(ax=ax, show_legend=True, color=LINE_COLOR,
                   label="Complex Impedance", draw_labels=True)
  for circle in ax.findobj(match=plt.Circle):
    if circle.get_edgecolor() != (0.0, 0.0, 0.0, 1.0):  # type: ignore
      circle.set_linestyle('dotted')                    # type: ignore
      circle.set_edgecolor("#444444")                   # type: ignore
    else:
      circle.set_edgecolor("#999999")                   # type: ignore


def vswr_plot(dut: rf.Network, ax: Axes) -> None:
  dut.plot_s_vswr(ax=ax, color=LINE_COLOR, label='VSWR')

  fmin = dut.frequency.f.min()
  fmax = dut.frequency.f.max()
  ax.set_xlim(fmin, fmax)
  ax.set_ylabel(r'$\rho$')

  max_vswr = (1+dut.s_mag.max())/(1-dut.s_mag.max())
  ax.set_ylim(1, 4 if max_vswr < 4 else max_vswr * 1.2 if max_vswr < 20 else 20)
  ax.axhline(y=2, linewidth=1, zorder=9, color=SWR_COLOR, linestyle="-.")
  ax.axhline(y=3, linewidth=1, zorder=9, color=SWR_COLOR, linestyle="-.")

  for low, high, _l in BANDS:
    ax.axvspan(low*1000, high*1000, facecolor='cyan', alpha=0.15)
    ax.legend(loc='upper right')

  freq = dut.frequency.f
  vswr = dut.s_vswr[:, 0, 0]
  peaks = signal.argrelextrema(vswr, lambda a, b: a < b, order=21)[0]
  if len(peaks) > 12:
    return

  text = []
  for idx in peaks:
    text.append(f'{fmt_freq(freq[idx])} VSWR: {vswr[idx]:.2f}')
    ax.annotate(f'{vswr[idx]:.2f}', xy=(freq[idx], vswr[idx]),
                xytext=(freq[idx], vswr[idx] + 0.5),
                ha='center', color=SWR_COLOR, fontsize=8,
                arrowprops={'facecolor': 'red', 'shrink': .005})

  if text:
    textstr = '\n'.join(text)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, zorder=20,
            fontsize=10, color=SWR_COLOR, verticalalignment='top')


def rl_plot(dut: rf.Network, ax: Axes) -> None:
  dut.plot_s_db(ax=ax, color=LINE_COLOR, label="Return Loss")

  fmin = dut.frequency.f.min()
  fmax = dut.frequency.f.max()

  ax.set_xlim(fmin, fmax)
  ax.set_ylim(top=0)
  ax.set_ylabel(r'$\Gamma$')
  ax.axhline(y=-6, linewidth=1, zorder=9, color=SWR_COLOR, linestyle="-.")
  ax.axhline(y=-10, linewidth=1, zorder=9, color=SWR_COLOR, linestyle="-.")
  for low, high, _ in BANDS:
    ax.axvspan(low*1000, high*1000, facecolor='cyan', alpha=0.15)
  ax.legend(loc='upper right')


def phase_plot(dut: rf.Network, ax: Axes) -> None:
  fmin = dut.frequency.f.min()
  fmax = dut.frequency.f.max()

  dut.plot_s_deg(ax=ax, color=LINE_COLOR, label="Phase")
  ax.set_xlim(fmin, fmax)
  ax.set_ylim(bottom=-180, top=180)
  ax.set_ylabel(r'$\phi$')
  for low, high, _ in BANDS:
    ax.axvspan(low*1000, high*1000, facecolor='lightgray', alpha=0.15)
  ax.legend(loc='upper right')


def impedance_plot(dut: rf.Network, ax: Axes) -> None:
  fmin = dut.frequency.f.min()
  fmax = dut.frequency.f.max()

  dut.plot_z_re(ax=ax, color=LINE_COLOR, label="| Z |")
  ax.set_xlim(fmin, fmax)
  ax.set_ylabel(r'$\Omega$')
  ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_ohm))
  for low, high, _ in BANDS:
    ax.axvspan(low*1000, high*1000, facecolor='lightgray', alpha=0.15)
  ax.legend(loc='upper center')


def dual_plot(dut: rf.Network, opts: argparse.Namespace) -> None:
  grid = GridSpec(2, 1, height_ratios=[3, 1])
  fig = plt.figure(figsize=(14, 14))
  fig.suptitle(opts.title)

  smith_chart(dut.s11, fig.add_subplot(grid[0, 0]))
  vswr_plot(dut.s11, fig.add_subplot(grid[1, 0]))

  fig.text(0.01, 0.01, __author__, color=LIGHTGRAY, fontsize=8)
  img_name = opts.target.joinpath(f'{slugify(opts.title)}-dual{opts.ext}')
  logging.info(img_name)
  fig.savefig(img_name, transparent=False)


def draw_all(dut: rf.Network, opts: argparse.Namespace) -> None:
  grid = GridSpec(2, 3, height_ratios=[2.25, 1])

  fig = plt.figure(figsize=(14, 14))
  fig.suptitle(opts.title)

  smith_chart(dut.s11, fig.add_subplot(grid[0, :]))
  vswr_plot(dut.s11, fig.add_subplot(grid[1, 0]))
  rl_plot(dut.s11, fig.add_subplot(grid[1, 1]))
  phase_plot(dut.s11, fig.add_subplot(grid[1, 2]))

  fig.text(0.01, 0.01, __author__, color=LIGHTGRAY, fontsize=8)
  img_name = opts.target.joinpath(f'{slugify(opts.title)}-all{opts.ext}')
  logging.info(img_name)
  fig.savefig(img_name, transparent=False)
  plt.close()


def draw(dut: rf.Network, call: Callable, opts: argparse.Namespace) -> None:
  if call is smith_chart:
    fig = plt.figure(figsize=(10, 10))
  else:
    fig = plt.figure()
  fig.suptitle(opts.title)
  call(dut.s11, fig.gca())

  fig.text(0.01, 0.01, __author__, color=LIGHTGRAY, fontsize=8)
  img_name = opts.target.joinpath(f'{slugify(opts.title)}-{call.__name__}{opts.ext}')
  logging.info(img_name)
  fig.savefig(img_name, transparent=False)
  plt.close()


def fmt_ohm(num: float, _: float | None = None) -> str:
  for unit in ("Ω", "KΩ", "MΩ", "GΩ"):
    if abs(num) < 1000.0:
      return f"{int(num):d}{unit}"
    num /= 1000.0
  return "∞"


def fmt_freq(num: float, _: float | None = None) -> str:
  for unit in ("Hz", "KHz", "MHz", "GHz", "THz"):
    if abs(num) < 1000.0:
      return f"{num:.1f}{unit}"
    num /= 1000.0
  return f"{num:.1f}PHz"


def type_range(parg) -> Tuple[float, float]:
  start, stop = [float(x) * 10**6 for x in parg.split(':')]
  return start, stop


def parse_args() -> argparse.Namespace:
  options = [('A', 'all'), ('D', 'dual'), ('v', 'vswr'), ('p', 'phase'),
             ('l', 'rloss'), ('s', 'smith'), ('i', 'impedance')]

  parser = argparse.ArgumentParser(
    prog='RigExpert',
    description=__doc__,
    epilog='This program can be found https://gist.github.com/0x9900',
    formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('-f', '--s1p-file', type=pathlib.Path, required=True)
  parser.add_argument('-e', '--ext', default='.png',
                      choices=[".png", ".jpg", ".svg", ".svgz", ".pdf", ".webp"])
  parser.add_argument('-t', '--title')
  parser.add_argument('-d', '--directory', dest='target', type=pathlib.Path, default='/tmp',
                      help=('Target directory where the images will be stored '
                            '(default: %(default)s)'))
  parser.add_argument('-r', '--range', type=type_range, help='Frequency range start:stop in MHz')
  d_group = parser.add_argument_group()
  for short, long in options:
    d_group.add_argument(f'-{short}', f'--{long}', action="store_true", default=False)
  opts = parser.parse_args()

  if not opts.title:
    opts.title = opts.s1p_file.stem

  if not opts.target.is_dir():
    parser.error(f'{opts.target} not found')

  if not any(getattr(opts, x) for _, x in options):
    parser.error(f'Choose one of the options {[f"--{o[1]}" for o in options]}')

  return opts


def main() -> None:
  functions = []
  plt.rcParams.update(MY_PARAMS)
  plt.tight_layout()

  opts = parse_args()

  try:
    dut = read_s1p(opts.s1p_file, opts.range)
  except (NotImplementedError, FileNotFoundError, ValueError) as err:
    raise SystemExit(err)

  if opts.all:
    draw_all(dut, opts)
  if opts.dual:
    dual_plot(dut, opts)

  if opts.vswr:
    functions.append(vswr_plot)
  if opts.phase:
    functions.append(phase_plot)
  if opts.rloss:
    functions.append(rl_plot)
  if opts.smith:
    functions.append(smith_chart)
  if opts.impedance:
    functions.append(impedance_plot)

  for function in functions:
    draw(dut, function, opts)


if __name__ == "__main__":
  main()
