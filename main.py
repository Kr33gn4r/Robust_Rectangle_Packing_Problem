import matplotlib
import numpy as np
from docplex.cp.model import CpoModel, CpoParameters
from typing import List, Tuple
from random import randint, random, choice
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from rectpack import newPacker

matplotlib.use('TkAgg')

"""
Rectangle Packing Problem
Find the smallest box by area in which the rectangles are bounded, rectangles cannot overlap.
"""

# ------------------------------------------------------------------------ #
# Data
def createRectangles(sizes: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Tuple[int, int]]:
    rects = []

    for rect in sizes:
        heights = rect[0]
        widths = rect[1]

        h_negative, h_positive = (heights[0], heights[1]) if heights[0] <= heights[1] else (heights[1], heights[0])
        w_negative, w_positive = (widths[0], widths[1]) if widths[0] <= widths[1] else (widths[1], widths[0])

        h_chosen = randint(h_negative, h_positive)
        w_chosen = randint(w_negative, w_positive)

        rects.append((h_chosen, w_chosen))
    return rects

# ------------------------------------------------------------------------ #
# Model
def solveRPP(rectangles: List[Tuple[int, int]]):
    heights = [rect[0] for rect in rectangles]
    widths = [rect[1] for rect in rectangles]

    with CpoModel(name="RPP") as mdl:
        mdl.set_parameters(CpoParameters(Workers=16,
                                         LogVerbosity="Normal",
                                         SearchType="Auto",
                                         DynamicProbingStrength=0.05,
                                         MultiPointNumberOfSearchPoints=30,
                                         TimeLimit=None,
                                         OptimalityTolerance=1e-4,
                                         RelativeOptimalityTolerance=1e-4))
        x = mdl.integer_var_list(len(rectangles), 0, 9007199254740991, name="x")
        y = mdl.integer_var_list(len(rectangles), 0, 9007199254740991, name="y")
        r = mdl.binary_var_list(len(rectangles), name="r")
        X = mdl.integer_var(name="XW")
        Y = mdl.integer_var(name="YH")

        for i in range(len(rectangles) - 1):
            for j in range(i + 1, len(rectangles)):
                mdl.add(
                    (x[i] + widths[i] * (1 - r[i]) + heights[i] * r[i] <= x[j]) |
                    (x[j] + widths[j] * (1 - r[j]) + heights[j] * r[j] <= x[i]) |
                    (y[i] + widths[i] * r[i] + heights[i] * (1 - r[i]) <= y[j]) |
                    (y[j] + widths[j] * r[j] + heights[j] * (1 - r[j]) <= y[i])
                )

        mdl.add(
            X == mdl.max(x[i] + widths[i] * (1 - r[i]) + heights[i] * r[i] for i in range(len(rectangles)))
        )

        mdl.add(
            Y == mdl.max(y[j] + widths[j] * r[j] + heights[j] * (1 - r[j]) for j in range(len(rectangles)))
        )

        mdl.add(mdl.minimize(mdl.log(X) + mdl.log(Y)))

        solution = mdl.solve()
        s = [(x.get_name(), x.get_value()) for x in solution.get_all_var_solutions()]

        Xs, Ys, Rs, XW, YH = [None] * len(rectangles), [None] * len(rectangles), [None] * len(rectangles), None, None
        for tup in s:
            if tup[0][0] == 'x':
                Xs[int(tup[0][2:])] = tup[1]
            elif tup[0][0] == 'y':
                Ys[int(tup[0][2:])] = tup[1]
            elif tup[0][0] == 'r':
                Rs[int(tup[0][2:])] = tup[1]
            elif tup[0][0] == 'X':
                XW = tup[1]
            else:
                YH = tup[1]

        return Xs, Ys, Rs, XW, YH

# ------------------------------------------------------------------------ #
# Fast Rectangle Packing Algorithm
def rectpacker(rectangles: List[Tuple[int, int]], cycles=25):
    BestXs, BestYs, BestRects, BestXW, BestYH = [], [], [], None, None
    BestRs = [1] * len(rectangles)

    for c in range(cycles):
        print(c)
        for i, r in enumerate(rectangles):
            if choice([True, False]):
                rectangles[i] = (r[1], r[0])

        packer = newPacker()
        for r in rectangles:
            packer.add_rect(*r)
        bin = (2000, 2000)
        packer.add_bin(*bin)
        packer.pack()

        Xs, Ys = [], []
        Rects = []

        for r in packer[0]:
            Xs.append(r.x)
            Ys.append(r.y)
            Rects.append((r.width, r.height))

        XW = max(x + y[0] for x, y in zip(Xs, Rects))
        YH = max(x + y[1] for x, y in zip(Ys, Rects))

        if (BestXW == None and BestYH == None or XW * YH < BestXW * BestYH):
            BestXs = Xs
            BestYs = Ys
            BestRects = Rects
            BestXW = XW
            BestYH = YH

    return BestXs, BestYs, BestRs, BestXW, BestYH, BestRects

# ------------------------------------------------------------------------ #
# Draw solution
def draw(X, Y, R, rects, XW, YH):
    fig, ax = plt.subplots()

    for x, y, r, (h, w) in zip(X, Y, R, rects):
        face_color = (random(), random(), random())
        if r:
            h, w = w, h
        rectangle = patches.Rectangle((x, y), w, h, facecolor=face_color, alpha=0.5)
        ax.add_patch(rectangle)

    ax.set_xlim(0, max(XW,YH))
    ax.set_ylim(0, max(XW,YH))

    plt.show()

# ------------------------------------------------------------------------ #
# Calculate minimal maximum regret

def minimax(ideal_areas, approx_areas):
    regrets = np.array(approx_areas) - np.array(ideal_areas)
    max_regret = np.max(regrets)

    max_regret_indices = np.where(regrets == max_regret)[0]
    max_regret_tuples = [(regrets[i], approx_areas[i], ideal_areas[i]) for i in max_regret_indices]

    minimum = min(max_regret_tuples, key=lambda x: x[2])

    print(minimum)
    return minimum

if __name__ == "__main__":
    data = [
        (
            (500, 550), (1000, 1100)
        ),
        (
            (440, 520), (600, 660)
        ),
        (
            (250, 300), (250, 300)
        ),
        (
            (500, 550), (660, 700)
        ),
        (
            (400, 450), (800, 880)
        )
    ]

    ideal_areas, approx_areas = [], []

    for _ in range(10):

        rectangles = createRectangles(data)

        Xs, Ys, Rs, XW, YH = solveRPP(rectangles)
        Xs2, Ys2, Rs2, XW2, YH2, rects = rectpacker(rectangles)

        ideal_areas.append(XW*YH)
        approx_areas.append(XW2*YH2)

        # draw(Xs, Ys, Rs, rectangles, XW, YH)
        # draw(Xs2, Ys2, Rs2, rects, XW2, YH2)

    minimax(ideal_areas, approx_areas)

