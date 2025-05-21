from datetime import timedelta
from typing import cast
import pandas as pd
import time


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
    return avg, median, distribution


def get_distribution_over_time(df: pd.DataFrame) -> pd.Series:
    # Get the first row per group (based on MultiIndex level 0)
    first_rows = df.groupby(level=0).first()

    # Extract year-month from the 'created_at' of the first row
    first_rows["creation_month"] = first_rows["created_at"].dt.to_period("M")

    # Count how many groups were created per month
    monthly_group_counts = first_rows["creation_month"].value_counts().sort_index()
    return monthly_group_counts


def main():
    start_time = time.time()
    df = cast(pd.DataFrame, pd.read_pickle("./conversations.pkl"))
    df["created_at"] = pd.to_datetime(df["created_at"])

    print(
        f"Number of conversations: {df.index.get_level_values('conversation').nunique()}"
    )
    print(f"Time taken: {str(timedelta(seconds=time.time() - start_time))}")
    avg, median, distribution = get_conv_stats(df)
    print(f"Average conversation length: {avg} tweets")
    print(f"Median conversation length: {median} tweets")
    mnth = get_distribution_over_time(df)


if __name__ == "__main__":
    main()
