r"""The Hershey single-stroke vector font (issue #11).

A pen plotter cannot draw a filled outline: it owns one pen tip and two axes,
so a letter has to *be* a path rather than a region to fill. That is what the
Hershey fonts are -- glyphs defined as the strokes a pen walks, which is why
they have been the plotter world's standard since the 1960s and why writing a
word here reduces to the waypoint lists `PenPlotter` already consumes.

The data below is the `futural` face (Hershey "simplex" sans, upper and lower
case), verbatim in James Hurt's JHF format, one glyph per line for ASCII 32
through 127:

    12345  9MWRFRT RRYQZR[SZRY
    \___/ \/ ||  \_______________ vertex pairs, one character per coordinate,
      |    |  \__ left/right      offset from 'R'; the pair " R" is a PEN LIFT
      |    \_ vertex count        bearings
      \_ glyph number (unused)

Coordinates are `ord(c) - ord('R')`, and JHF's y axis points DOWN with the
baseline at y = +9. This module flips it once, at parse time, into the frame
the rest of the repo thinks in -- x right, **y up**, origin on the baseline at
the glyph's left bearing -- so no consumer has to remember the flip. The
board's own lateral sign flip lives in `tools/strokes.py`, for the same reason:
each convention gets exactly one place where it is applied.

Sizes are quoted as CAP HEIGHT (the height of an 'A'), not as an em, because
cap height is the number you can actually measure on a photograph of a board.

  The Hershey Fonts were originally created by Dr. A. V. Hershey while working
  at the U. S. National Bureau of Standards. The format of the font data in
  this distribution was originally created by James Hurt, Cognition, Inc.
"""

from dataclasses import dataclass

FIRST_CHAR = " "          # the JHF table starts at ASCII 32

_FUTURAL = r"""
12345  1JZ
12345  9MWRFRT RRYQZR[SZRY
12345  6JZNFNM RVFVM
12345 12H]SBLb RYBRb RLOZO RKUYU
12345 27H\PBP_ RTBT_ RYIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX
12345 32F^[FI[ RNFPHPJOLMMKMIKIIJGLFNFPGSHVHYG[F RWTUUTWTYV[X[ZZ[X[VYTWT
12345 35E_\O\N[MZMYNXPVUTXRZP[L[JZIYHWHUISJRQNRMSKSIRGPFNGMIMKNNPQUXWZY[[[\Z\Y
12345  8MWRHQGRFSGSIRKQL
12345 11KYVBTDRGPKOPOTPYR]T`Vb
12345 11KYNBPDRGTKUPUTTYR]P`Nb
12345  9JZRLRX RMOWU RWOMU
12345  6E_RIR[ RIR[R
12345  8NVSWRXQWRVSWSYQ[
12345  3E_IR[R
12345  6NVRVQWRXSWRV
12345  3G][BIb
12345 18H\QFNGLJKOKRLWNZQ[S[VZXWYRYOXJVGSFQF
12345  5H\NJPISFS[
12345 15H\LKLJMHNGPFTFVGWHXJXLWNUQK[Y[
12345 16H\MFXFRNUNWOXPYSYUXXVZS[P[MZLYKW
12345  7H\UFKTZT RUFU[
12345 18H\WFMFLOMNPMSMVNXPYSYUXXVZS[P[MZLYKW
12345 24H\XIWGTFRFOGMJLOLTMXOZR[S[VZXXYUYTXQVOSNRNOOMQLT
12345  6H\YFO[ RKFYF
12345 30H\PFMGLILKMMONSOVPXRYTYWXYWZT[P[MZLYKWKTLRNPQOUNWMXKXIWGTFPF
12345 24H\XMWPURRSQSNRLPKMKLLINGQFRFUGWIXMXRWWUZR[P[MZLX
12345 12NVROQPRQSPRO RRVQWRXSWRV
12345 14NVROQPRQSPRO RSWRXQWRVSWSYQ[
12345  4F^ZIJRZ[
12345  6E_IO[O RIU[U
12345  4F^JIZRJ[
12345 21I[LKLJMHNGPFTFVGWHXJXLWNVORQRT RRYQZR[SZRY
12345 56E`WNVLTKQKOLNMMPMSNUPVSVUUVS RQKOMNPNSOUPV RWKVSVUXVZV\T]Q]O\L[JYHWGTFQFNGLHJJILHOHRIUJWLYNZQ[T[WZYYZX RXKWSWUXV
12345  9I[RFJ[ RRFZ[ RMTWT
12345 24G\KFK[ RKFTFWGXHYJYLXNWOTP RKPTPWQXRYTYWXYWZT[K[
12345 19H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZV
12345 16G\KFK[ RKFRFUGWIXKYNYSXVWXUZR[K[
12345 12H[LFL[ RLFYF RLPTP RL[Y[
12345  9HZLFL[ RLFYF RLPTP
12345 23H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZVZS RUSZS
12345  9G]KFK[ RYFY[ RKPYP
12345  3NVRFR[
12345 11JZVFVVUYTZR[P[NZMYLVLT
12345  9G\KFK[ RYFKT RPOY[
12345  6HYLFL[ RL[X[
12345 12F^JFJ[ RJFR[ RZFR[ RZFZ[
12345  9G]KFK[ RKFY[ RYFY[
12345 22G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF
12345 14G\KFK[ RKFTFWGXHYJYMXOWPTQKQ
12345 25G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF RSWY]
12345 17G\KFK[ RKFTFWGXHYJYLXNWOTPKP RRPY[
12345 21H\YIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX
12345  6JZRFR[ RKFYF
12345 11G]KFKULXNZQ[S[VZXXYUYF
12345  6I[JFR[ RZFR[
12345 12F^HFM[ RRFM[ RRFW[ R\FW[
12345  6H\KFY[ RYFK[
12345  7I[JFRPR[ RZFRP
12345  9H\YFK[ RKFYF RK[Y[
12345 12KYOBOb RPBPb ROBVB RObVb
12345  3KYKFY^
12345 12KYTBTb RUBUb RNBUB RNbUb
12345  6JZRDJR RRDZR
12345  3I[Ib[b
12345  8NVSKQMQORPSORNQO
12345 18I\XMX[ RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
12345 18H[LFL[ RLPNNPMSMUNWPXSXUWXUZS[P[NZLX
12345 15I[XPVNTMQMONMPLSLUMXOZQ[T[VZXX
12345 18I\XFX[ RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
12345 18I[LSXSXQWOVNTMQMONMPLSLUMXOZQ[T[VZXX
12345  9MYWFUFSGRJR[ ROMVM
12345 23I\XMX]W`VaTbQbOa RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
12345 11I\MFM[ RMQPNRMUMWNXQX[
12345  9NVQFRGSFREQF RRMR[
12345 12MWRFSGTFSERF RSMS^RaPbNb
12345  9IZMFM[ RWMMW RQSX[
12345  3NVRFR[
12345 19CaGMG[ RGQJNLMOMQNRQR[ RRQUNWMZM\N]Q][
12345 11I\MMM[ RMQPNRMUMWNXQX[
12345 18I\QMONMPLSLUMXOZQ[T[VZXXYUYSXPVNTMQM
12345 18H[LMLb RLPNNPMSMUNWPXSXUWXUZS[P[NZLX
12345 18I\XMXb RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
12345  9KXOMO[ ROSPPRNTMWM
12345 18J[XPWNTMQMNNMPNRPSUTWUXWXXWZT[Q[NZMX
12345  9MYRFRWSZU[W[ ROMVM
12345 11I\MMMWNZP[S[UZXW RXMX[
12345  6JZLMR[ RXMR[
12345 12G]JMN[ RRMN[ RRMV[ RZMV[
12345  6J[MMX[ RXMM[
12345 10JZLMR[ RXMR[P_NaLbKb
12345  9J[XMM[ RMMXM RM[X[
12345 40KYTBRCQDPFPHQJRKSMSOQQ RRCQEQGRISJTLTNSPORSTTVTXSZR[Q]Q_Ra RQSSUSWRYQZP\P^Q`RaTb
12345  3NVRBRb
12345 40KYPBRCSDTFTHSJRKQMQOSQ RRCSESGRIQJPLPNQPURQTPVPXQZR[S]S_Ra RSSQUQWRYSZT\T^S`RaPb
12345 24F^IUISJPLONOPPTSVTXTZS[Q RISJQLPNPPQTTVUXUZT[Q[O
12345 35JZJFJ[K[KFLFL[M[MFNFN[O[OFPFP[Q[QFRFR[S[SFTFT[U[UFVFV[W[WFXFX[Y[YFZFZ[
"""


Polyline = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Glyph:
  """One character's strokes, in font units, x right and y up.

  `advance` is how far the pen origin moves to set the next character; it
  already contains the side bearings the font designer chose, so laying text
  out is a running sum and nothing else.
  """
  char: str
  advance: float
  strokes: tuple[Polyline, ...]

  @property
  def ink_width(self) -> float:
    """Width of the marks themselves, which is NOT `advance` -- a space has an
    advance and no ink at all."""
    xs = [x for s in self.strokes for x, _ in s]
    return (max(xs) - min(xs)) if xs else 0.0


def _parse_glyph(char: str, line: str) -> Glyph:
  count = int(line[5:8])
  body = line[8:]
  left, right = ord(body[0]) - ord("R"), ord(body[1]) - ord("R")
  strokes: list[list[tuple[float, float]]] = []
  run: list[tuple[float, float]] = []
  for k in range(1, count):
    cx, cy = body[2 * k], body[2 * k + 1]
    if cx == " " and cy == "R":          # pen up: end the current stroke
      if run:
        strokes.append(run)
        run = []
      continue
    # x relative to the left bearing; y flipped about the baseline (y = 9).
    run.append((float(ord(cx) - ord("R") - left), float(9 - (ord(cy) - ord("R")))))
  if run:
    strokes.append(run)
  return Glyph(char, float(right - left),
               tuple(tuple(s) for s in strokes))


def _parse_font(raw: str) -> dict[str, Glyph]:
  glyphs = {}
  for i, line in enumerate(ln for ln in raw.splitlines() if ln.strip()):
    glyphs[chr(ord(FIRST_CHAR) + i)] = _parse_glyph(chr(ord(FIRST_CHAR) + i), line)
  return glyphs


GLYPHS: dict[str, Glyph] = _parse_font(_FUTURAL)

# Measured from the table rather than asserted about it: an 'A' spans the cap
# height by definition, and the deepest descender sets how far a line of text
# reaches below its own baseline. Both matter for fitting text to a board, and
# both would be a lie if someone swapped the face for a different JHF file.
CAP_UNITS = max(y for s in GLYPHS["A"].strokes for _, y in s)
DESCENDER_UNITS = min(y for c in "gjpqy" for s in GLYPHS[c].strokes for _, y in s)
MISSING = "?"             # what an unsupported character draws as


def glyph(char: str) -> Glyph:
  """The glyph for `char`, falling back to '?' rather than raising.

  Text on this board comes from an LLM eventually (issue #15), and a single
  unlucky character is not a reason to abandon a whole errand -- a visible '?'
  is a better failure than a traceback three minutes into a drawing.
  """
  return GLYPHS.get(char, GLYPHS[MISSING])


def supported(text: str) -> bool:
  """Whether every character of `text` has a real glyph (spaces included)."""
  return all(c in GLYPHS for c in text)


def text_width(text: str, cap_height: float, tracking: float = 0.0) -> float:
  """Width of a single line, in metres. Layout is a running sum of advances,
  so this is exact rather than an estimate."""
  if not text:
    return 0.0
  scale = cap_height / CAP_UNITS
  return (sum(glyph(c).advance for c in text) * scale
          + tracking * (len(text) - 1))


FIT_TOL = 1e-9            # relative slack in "does this line fit?".
                          # `strokes.fit_text` shrinks text until the widest
                          # word is EXACTLY the line width, so an exact
                          # comparison decides that word by float rounding and
                          # hard-breaks it: measured, "GOOD MORNING" shrunk to
                          # fit came out as "GOOD / MORNIN / G". The tolerance
                          # belongs in the comparison, not in a fudged cap
                          # height -- the question being asked is "does it
                          # fit", and 1 part in 10^9 of a 100 mm line is 0.1 nm.


def _fits(width: float, max_width: float) -> bool:
  return width <= max_width * (1 + FIT_TOL)


def wrap(text: str, cap_height: float, max_width: float,
         tracking: float = 0.0) -> list[str]:
  """Greedy word wrap to `max_width` metres.

  A word longer than the whole line gets HARD-BROKEN rather than allowed to
  run off the board: the bounds check downstream is a hard machine limit
  (the carriage has 110 mm of travel and no more), so silently overflowing
  would only move the failure to a place with less context.
  """
  lines: list[str] = []
  for paragraph in text.split("\n"):
    line = ""
    for word in paragraph.split():
      trial = f"{line} {word}" if line else word
      if line and not _fits(text_width(trial, cap_height, tracking), max_width):
        lines.append(line)
        line = word
      else:
        line = trial
      while len(line) > 1 and not _fits(
          text_width(line, cap_height, tracking), max_width):
        cut = len(line)
        while cut > 1 and not _fits(
            text_width(line[:cut], cap_height, tracking), max_width):
          cut -= 1
        lines.append(line[:cut])
        line = line[cut:]
    lines.append(line)
  return lines


def layout(text: str, cap_height: float = 0.020, max_width: float | None = None,
           line_gap: float | None = None, tracking: float = 0.0,
           align: str = "center") -> list[Polyline]:
  """Render `text` to polylines in metres: x right, y up, block centred on the
  origin.

  Centred on the origin because that is what the plotter wants: `draw` centres
  a figure on where the pen actually is, not on the middle of the board.
  """
  scale = cap_height / CAP_UNITS
  gap = line_gap if line_gap is not None else cap_height * 1.7
  lines = wrap(text, cap_height, max_width, tracking) if max_width else \
      text.split("\n")
  widths = [text_width(ln, cap_height, tracking) for ln in lines]
  block_w = max(widths) if widths else 0.0
  # Vertical centring uses the BASELINE band plus the descender depth, so a
  # word with a 'g' in it sits the same as one without: the reader sees the
  # line of text centred, not the ink's bounding box.
  desc = -DESCENDER_UNITS * scale
  block_h = cap_height + desc + gap * (len(lines) - 1)
  top = block_h / 2 - cap_height

  out: list[Polyline] = []
  for row, line in enumerate(lines):
    if align == "center":
      pen_x = -widths[row] / 2
    elif align == "right":
      pen_x = block_w / 2 - widths[row]
    else:
      pen_x = -block_w / 2
    pen_y = top - row * gap
    for char in line:
      g = glyph(char)
      for stroke in g.strokes:
        out.append(tuple((pen_x + x * scale, pen_y + y * scale)
                         for x, y in stroke))
      pen_x += g.advance * scale + tracking
  return out
