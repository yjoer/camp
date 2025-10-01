# %%
import pandas as pd

# %%
df = pd.DataFrame(
    {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
        "C": [1.0, 2.0, 3.0],
        "D": [4.0, 5.0, 6.0],
        "E": ["A", "B", "C"],
        "F": ["D", "E", "F"],
    },
)

# %%
df._mgr

# %%
with pd.option_context("mode.copy_on_write", False):  # noqa: FBT003
    df_copy = df.copy()
    vw = df_copy[:]
    vw.iloc[0, 0] = 10

    print(df_copy._mgr)
    print(vw._mgr)

# %%
with pd.option_context("mode.copy_on_write", True):  # noqa: FBT003
    vw = df[:]
    vw.iloc[0, 0] = 10

    print(df._mgr)
    print(vw._mgr)

# %%
