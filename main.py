import sys
import cv2
import numpy as np

photo = []
tokens = []
gaps = []
model = []


def read():
    global photo

    path = ""
    args = sys.argv[1:]

    if len(args) >= 2:
        path = args[1]

    photo = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    return


def delete_outliers(arr):
    res = []

    m, s = np.array(arr).mean(), np.array(arr).std()

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
    maxr = 0
    height_threshold = find_height_threshold(photo)

    l, r = photo.shape

    for i in range(r):
        e = 0
        for k in range(l):
            if photo[k][i] == 255:
                e = e + 1
                if e > height_threshold:
                    maxr = max(k, maxr)
                    break
            else:
                e = 0

    if maxr < height_threshold:
        maxr = l

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
    global tokens

    last_block = 0
    e = 0

    row_threshold = find_row_threshold(photo)

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
        if e > row_threshold:
            tokens.append(photo[:, last_block:i + 1])
            last_block = i + 1
            e = 0

    if last_block < r:
        tokens.append(photo[:, last_block:])

    return


def tokenize():
    global photo, tokens
    while len(photo) != 0:
        photo = normalize(photo)
        lh = find_line_height(np.array(photo))
        cut = photo[:lh + 1, :]
        add_row_tokens(cut)
        photo = photo[lh + 1:, :]

    return


def is_white(token_photo):
    l, r = token_photo.shape

    for i in range(l):
        for j in range(r):
            if token_photo[i][j] != 255:
                return False

    return True


def filter_tokens():
    global tokens
    filtered_tokens = []
    for i in tokens:
        if not is_white(i):
            filtered_tokens.append(i)

    tokens = filtered_tokens

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
    filter_tokens()
    train()
    find_Answer()


run()
