import os

from tensorflow.python.keras.utils.np_utils import to_categorical


def calculate_changes(df):
    "add a column containing the percent change or Opening price (first value is 0)"
    changes = []
    for i in range(len(df)):
        if i == 0:
            changes.append(0)
            continue
        change = (df["Open"][i] - df["Open"][i - 1]) / df["Open"][i - 1]
        changes.append(change)
    df["% change"] = changes
    return


def calc_net_change(arr):
  change = 0
  for y in range(len(arr)):
      change += 1
      change = change*(1 + arr[y])
      change -= 1
  return change
