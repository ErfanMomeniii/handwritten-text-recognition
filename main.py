import sys
import cv2
import numpy as np

photo = []
tokens = []
gaps = []
model = []

height_threshold_pixel = None
row_threshold_pixel = None


def read():
    global photo
    args = sys.argv[1:]
    path = ""
    if len(args) >= 2:
        path = args[1]
    photo = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return


def delete_outliers(arr):
    m, s = np.array(arr).mean(), np.array(arr).std()
    res = []
    for i in arr:
        if abs(i - m) <= s:
            res.append(i)
    return res


def find_height_threshold(photo):
    gaps = []
    l, r = photo.shape
    for i in range(r):
        e = 0
        for j in range(l):
            if photo[j][i] == 255:
                e = e + 1
            else:
                if e != 0:
                    gaps.append(e)
                e = 0
        if e != 0:
            gaps.append(e)
    gaps = delete_outliers(gaps)
    return np.array(gaps).mean()


def find_row_threshold(photo):
    gaps = []
    l, r = photo.shape
    for i in range(l):
        e = 0
        for j in range(r):
            if photo[i][j] == 255:
                e = e + 1
            else:
                if e != 0:
                    gaps.append(e)
                e = 0
        if e != 0:
            gaps.append(e)
    gaps = delete_outliers(gaps)
    return np.array(gaps).mean()


def delete_first_row(photo):
    _, r = photo.shape
    for p in range(r):
        if photo[0][p] != 255:
            return False

    return True


def delete_last_row(photo):
    l, r = photo.shape
    for p in range(r):
        if photo[l - 1][p] != 255:
            return False

    return True


def delete_first_column(photo):
    l, _ = photo.shape
    for p in range(l):
        if photo[p][0] != 255:
            return False

    return True


def delete_last_column(photo):
    l, r = photo.shape
    for p in range(l):
        if photo[p][r - 1] != 255:
            return False

    return True


def normalize(photo):
    photo[photo > 127] = 255
    while delete_first_row(photo):
        photo = photo[1:, :]
    while delete_last_row(photo):
        photo = photo[:-1, :]
    while delete_first_column(photo):
        photo = photo[:, 1:]
    while delete_last_column(photo):
        photo = photo[:, :-1]
    return photo


def find_line_height(photo):
    global height_threshold_pixel
    height_threshold_pixel = find_height_threshold(photo)
    maxr = 0
    r, l = photo.shape
    for i in range(l):
        e = 0
        for k in range(r):
            if photo[k][i] == 255:
                e = e + 1
                if e > height_threshold_pixel:
                    maxr = max(k, maxr)
                    break
            else:
                e = 0
    if maxr < height_threshold_pixel:
        maxr = r
    return maxr


def left_trim(photo):
    e = 0
    l, r = photo.shape

    for i in range(r):
        has_black = False
        for j in range(l):
            if photo[j][i] != 255:
                has_black = True
                break
        if not has_black:
            e = e + 1
        else:
            return photo[:, e:]
    return []


def right_trim(photo):
    e = 0
    l, r = photo.shape

    for i in range(r):
        has_black = False
        for j in range(l):
            if photo[j][r - i - 1] != 255:
                has_black = True
                break
        if not has_black:
            e = e + 1
        else:
            return photo[:, :r - e + 1]
    return []


def add_row_tokens(photo):
    global row_threshold_pixel, tokens
    row_threshold_pixel = find_row_threshold(photo)
    lastblock = 0
    e = 0

    photo = left_trim(photo)
    photo = right_trim(photo)

    l, r = photo.shape

    for i in range(r):
        has_black = False
        for j in range(l):
            if photo[j][i] != 255:
                has_black = True
                break
        if not has_black:
            e = e + 1
        else:
            e = 0
        if e > row_threshold_pixel:
            tokens.append(photo[:, lastblock:i + 1])
            lastblock = i + 1
            e = 0
    if lastblock < r:
        tokens.append(photo[:, lastblock:])


def tokenize():
    global photo, tokens
    while len(photo) != 0:
        photo = normalize(photo)
        lh = find_line_height(np.array(photo))
        cut = photo[:lh + 1, :]
        add_row_tokens(cut)
        photo = photo[lh + 1:, :]
    return


def train():
    global tokens
    for i in range(len(tokens)):
        cv2.imshow("c", tokens[i])
        cv2.waitKey(0)
    return


def find_Answer():
    return


def run():
    read()
    tokenize()
    train()
    find_Answer()


run()
