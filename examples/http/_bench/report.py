# %%
import altair as alt
import pandas as pd

# %%
df = pd.read_json(".build/results.json", lines=True)
df_expanded = df.explode("measurements")

# %%
server_order = df["server_name"].unique().tolist()

# %%
rps_chart = (
  alt.Chart(df_expanded, title="Requests per Second")
  .mark_circle(size=24)
  .encode(
    x=alt.X(
      shorthand="server_name:N",
      axis=alt.Axis(grid=True, gridOpacity=0.25, labelAngle=-45),
      sort=server_order,
    ),
    y=alt.Y("measurements:Q", scale=alt.Scale(padding=15, zero=False)),
    xOffset="jitter:Q",
    color=alt.Color("language:N"),
  )
  .transform_calculate(jitter="random()")
  .properties(width=300)
)

# %%
latency_chart = (
  alt.Chart(df_expanded, title="Latency (μs)")
  .mark_circle(size=24)
  .encode(
    x=alt.X(
      shorthand="server_name:N",
      axis=alt.Axis(grid=True, gridOpacity=0.25, labelAngle=-45),
      sort=server_order,
    ),
    y=alt.Y("latency_us:Q", scale=alt.Scale(padding=15, zero=False)),
    xOffset="jitter:Q",
    color=alt.Color("language:N"),
  )
  .transform_calculate(
    jitter="random()",
    latency_us="1000000 / datum.measurements",
  )
  .properties(width=300)
)

# %%
(rps_chart | latency_chart).configure_title(fontSize=14).configure_axis(
  labelFontSize=14,
  titleFontSize=14,
).configure_legend(
  labelFontSize=14,
  titleFontSize=14,
)

# %%
df_summary = df[["server_name"]].rename(columns={"server_name": ""})

five_ = lambda x: pd.Series(x[:5])
df_summary[["1", "2", "3", "4", "5"]] = df["measurements"].apply(five_)

df_summary["x̄"] = df["mean_rps"]
df_summary["σ"] = df["std_rps"]

print(df_summary.to_markdown(index=False, floatfmt=".2f"))

# %%
df_latency = df[["server_name"]].rename(columns={"server_name": ""})

df_latency["t/µs"] = df["mean_latency_us"]
df_latency = df_latency.sort_values("t/µs")

min_latency = df["mean_latency_us"].min()
df_latency["%"] = min_latency / df_latency["t/µs"]

min_latency_by_lang = df.groupby("language")["mean_latency_us"].transform("min")
df_latency["%%"] = min_latency_by_lang / df_latency["t/µs"]

print(df_latency.to_markdown(index=False, floatfmt=("", ".2f", ".4f", ".4f")))

# %%
