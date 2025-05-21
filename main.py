from datetime import timedelta
from typing import cast
import pandas as pd
import time

pd.set_option("display.max_rows", 500)


def compute_difference(group):
    first_row = group.iloc[0]["sentiment_score"]
    if len(group) < 3:
        avg_rest = 0
    else:
        avg_rest = group.iloc[2:]["sentiment_score"].mean()
    return first_row - avg_rest


def get_conv_stats(df: pd.DataFrame) -> tuple[float, float, pd.Series]:
    sizes = df.groupby(level=0).size()
    avg = sizes.mean()
    median = sizes.median()
    distribution = sizes.value_counts().sort_values(ascending=False)
    print(distribution)
    return avg, median, distribution


def main():
    start_time = time.time()
    df = cast(pd.DataFrame, pd.read_pickle("./conversations.pkl"))
    print(
        f"Number of conversations: {df.index.get_level_values('conversation').nunique()}"
    )
    print(f"Time taken: {str(timedelta(seconds=time.time() - start_time))}")
    avg, median, _ = get_conv_stats(df)
    print(f"Average conversation length: {avg} tweets")
    print(f"Median conversation length: {median} tweets")


if __name__ == "__main__":
    main()
