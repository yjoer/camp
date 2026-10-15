# %%
import altair as alt
import pandas as pd

# %%
df = pd.read_json(".build/postgresql_results.jsonl", lines=True)
df["category"] = df["runner"] + df["port"].map({5432: "", 6432: " + pgbouncer"})

# %%
category_order = ["node", "pgbench", "node + pgbouncer", "pgbench + pgbouncer"]
df["category"] = pd.Categorical(df["category"], categories=category_order, ordered=True)
df = df.sort_values(["clients", "category"])


# %% [markdown]
# ## Throughput

# %%
def throughput_chart(title: str, data: pd.DataFrame) -> alt.Chart:
	base = alt.Chart(data, title=title).encode(
		x=alt.X("clients:O", axis=alt.Axis(labelAngle=-45)),
		xOffset=alt.X("category:N", sort=category_order),
		color=alt.Color("category:N", sort=category_order),
	)
	bars = base.mark_bar().encode(
		y=alt.Y("mean_rate:Q", stack=None),
		tooltip=["category", "clients", "mean_rate", "std_rate"],
	)
	errors = base.mark_errorbar().encode(
		y=alt.Y("mean_rate:Q", stack=None),
		yError="std_rate:Q",
	)
	return (bars + errors).properties(width=640, height=320)


# %%
rate_chart_sm = throughput_chart("Throughput (5-100 Δ5)", df[df["clients"] <= 100])
rate_chart_lg = throughput_chart("Throughput (100-1000 Δ100)", df[df["clients"] >= 100])

# %%
rate_chart_sm.configure_title(fontSize=14).configure_axis(
	labelFontSize=14,
	titleFontSize=14,
).configure_legend(
	labelFontSize=14,
	titleFontSize=14,
)

# %%
rate_chart_lg.configure_title(fontSize=14).configure_axis(
	labelFontSize=14,
	titleFontSize=14,
).configure_legend(
	labelFontSize=14,
	titleFontSize=14,
)


# %% [markdown]
# ## Latency

# %%
def latency_chart(title: str, data: pd.DataFrame) -> alt.Chart:
	base = alt.Chart(data, title=title).encode(
		x=alt.X("clients:O", axis=alt.Axis(labelAngle=-45)),
		xOffset=alt.X("category:N", sort=category_order),
		color=alt.Color("category:N", sort=category_order),
	)
	bars = base.mark_bar().encode(
		y=alt.Y("mean_latency_ms:Q", stack=None),
		tooltip=["category", "clients", "mean_latency_ms", "std_latency_ms"],
	)
	errors = base.mark_errorbar().encode(
		y=alt.Y("mean_latency_ms:Q", stack=None),
		yError="std_latency_ms:Q",
	)
	return (bars + errors).properties(width=640, height=320)


# %%
latency_chart_sm = latency_chart("Latency (5-100 Δ5)", df[df["clients"] <= 100])
latency_chart_lg = latency_chart("Latency (100-1000 Δ100)", df[df["clients"] >= 100])

# %%
latency_chart_sm.configure_title(fontSize=14).configure_axis(
	labelFontSize=14,
	titleFontSize=14,
).configure_legend(
	labelFontSize=14,
	titleFontSize=14,
)

# %%
latency_chart_lg.configure_title(fontSize=14).configure_axis(
	labelFontSize=14,
	titleFontSize=14,
).configure_legend(
	labelFontSize=14,
	titleFontSize=14,
)


# %% [markdown]
# ## Pooling Delta

# %%
def gain_loss_levels(frame: pd.DataFrame) -> pd.DataFrame:
	gain = frame[frame["delta_pct"] > 0].groupby("runner")["delta_pct"].mean().rename("gain")
	loss = frame[frame["delta_pct"] < 0].groupby("runner")["delta_pct"].mean().rename("loss")
	return pd.concat([gain, loss], axis=1).reset_index().melt(
		id_vars="runner",
		value_vars=["gain", "loss"],
		var_name="outcome",
		value_name="level",
	)


# %%
direct = df[df["port"] == 5432].set_index(["runner", "clients"])["mean_rate"]
pooled = df[df["port"] == 6432].set_index(["runner", "clients"])["mean_rate"]
delta = pd.DataFrame({
	"direct_rate": direct,
	"pooled_rate": pooled,
	"delta_pct": (pooled - direct) / direct * 100,
}).reset_index()
delta_levels = gain_loss_levels(delta)

# %%
delta_chart = (
	alt.Chart(delta, title="Pooling Delta (%)")
	.mark_bar()
	.encode(
		x=alt.X("clients:O", axis=alt.Axis(labelAngle=-45)),
		y=alt.Y("delta_pct:Q", stack=None),
		xOffset="runner:N",
		color="runner:N",
		tooltip=["runner", "clients", "direct_rate", "pooled_rate", "delta_pct"],
	)
	+ alt.Chart(delta_levels).mark_rule(opacity=0.2).encode(
		y="level:Q",
		color="runner:N",
		strokeDash="outcome:N",
	)
	+ alt.Chart(delta_levels).mark_rule(opacity=0, strokeWidth=12).encode(
		y="level:Q",
		color="runner:N",
		tooltip=["runner", "outcome", "level"],
	)
).properties(width=640, height=320)

# %%
delta_chart.configure_title(fontSize=14).configure_axis(
	labelFontSize=14,
	titleFontSize=14,
).configure_legend(
	labelFontSize=14,
	titleFontSize=14,
)

# %%
direct_latency = df[df["port"] == 5432].set_index(["runner", "clients"])["mean_latency_ms"]
pooled_latency = df[df["port"] == 6432].set_index(["runner", "clients"])["mean_latency_ms"]
delta_latency = pd.DataFrame({
	"direct_latency": direct_latency,
	"pooled_latency": pooled_latency,
	"delta_pct": (pooled_latency - direct_latency) / direct_latency * 100,
}).reset_index()
delta_latency_levels = gain_loss_levels(delta_latency)

# %%
delta_latency_chart = (
	alt.Chart(delta_latency, title="Pooling Delta Latency (%)")
	.mark_bar()
	.encode(
		x=alt.X("clients:O", axis=alt.Axis(labelAngle=-45)),
		y=alt.Y("delta_pct:Q", stack=None),
		xOffset="runner:N",
		color="runner:N",
		tooltip=["runner", "clients", "direct_latency", "pooled_latency", "delta_pct"],
	)
	+ alt.Chart(delta_latency_levels).mark_rule(opacity=0.2).encode(
		y="level:Q",
		color="runner:N",
		strokeDash="outcome:N",
	)
	+ alt.Chart(delta_latency_levels).mark_rule(opacity=0, strokeWidth=12).encode(
		y="level:Q",
		color="runner:N",
		tooltip=["runner", "outcome", "level"],
	)
).properties(width=640, height=320)

# %%
delta_latency_chart.configure_title(fontSize=14).configure_axis(
	labelFontSize=14,
	titleFontSize=14,
).configure_legend(
	labelFontSize=14,
	titleFontSize=14,
)


# %% [markdown]
# ## Throughput Summary

# %%
def summary_table(data: pd.DataFrame) -> None:
	table = data[["runner", "clients"]].rename(columns={"clients": "n"})

	five_ = lambda x: pd.Series(x[:5])
	table[["1", "2", "3", "4", "5"]] = data["rate_samples"].apply(five_)

	table["x̄"] = data["mean_rate"].to_numpy()
	table["σ"] = data["std_rate"].to_numpy()

	print(table.to_markdown(index=False, floatfmt=".2f"))


# %%
summary_table(df[df["port"] == 5432])

# %%
summary_table(df[df["port"] == 6432])


# %% [markdown]
# ## Latency Summary

# %%
def latency_table(data: pd.DataFrame) -> None:
	table = data[["runner", "clients"]].rename(columns={"clients": "n"})

	table["t/ms"] = data["mean_latency_ms"].to_numpy()
	table = table.sort_values("t/ms")

	min_latency = data["mean_latency_ms"].min()
	table["%"] = min_latency / table["t/ms"]

	print(table.to_markdown(index=False, floatfmt=("", "", ".2f", ".4f")))


# %%
latency_table(df[df["port"] == 5432])

# %%
latency_table(df[df["port"] == 6432])

# %%
