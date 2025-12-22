from grafana_foundation_sdk.builders.common import VizLegendOptions
from grafana_foundation_sdk.builders.dashboard import Dashboard
from grafana_foundation_sdk.builders.dashboard import QueryVariable
from grafana_foundation_sdk.builders.dashboard import Row
from grafana_foundation_sdk.builders.prometheus import Dataquery as PrometheusQuery
from grafana_foundation_sdk.builders.timeseries import Panel as Timeseries
from grafana_foundation_sdk.cog.encoder import JSONEncoder
from grafana_foundation_sdk.models.common import LegendDisplayMode
from grafana_foundation_sdk.models.common import LegendPlacement
from grafana_foundation_sdk.models.common import TimeZoneBrowser
from grafana_foundation_sdk.models.dashboard import VariableOption
from grafana_foundation_sdk.models.resource import Manifest
from grafana_foundation_sdk.models.resource import Metadata


def dashboard() -> Dashboard:
  builder = (
    Dashboard("SMARTctl Exporter")
    .uid("smartctl-exporter")
    .tags(["prometheus"])
    .refresh("5s")
    .time("now-24h", "now")
    .timezone(TimeZoneBrowser)
  )

  builder = variables(builder)

  builder = builder.with_row(Row("Overview"))
  builder = disk_temperature(builder)

  builder = builder.with_row(Row("Pre-Fail"))
  builder = raw_read_error_rate(builder)
  builder = spin_up_time(builder)
  builder = reallocated_sector_count(builder)

  row = Row("Old Age")
  row = start_stop_count(row)
  row = seek_error_rate(row)
  row = power_on_hours(row)
  row = spin_retry_count(row)
  row = calibration_retry_count(row)
  row = power_cycle_count(row)
  row = power_off_retract_count(row)
  row = load_cycle_count(row)
  row = reallocated_event_count(row)
  row = current_pending_sector_count(row)
  row = offline_uncorrectable(row)
  row = udma_crc_error_count(row)
  row = multi_zone_error_rate(row)

  return builder.with_row(row)


def variables(builder: Dashboard) -> Dashboard:
  return (
    builder.with_variable(
      QueryVariable("node")
      .label("node")
      .query("label_values(smartctl_version, instance)")
      .current(VariableOption(selected=True, text="All", value="$__all"))
      .include_all(include_all=True)
      .multi(multi=True),
    )
    .with_variable(
      QueryVariable("disk")
      .label("disk")
      .query("label_values(smartctl_device, device)")
      .current(VariableOption(selected=True, text="All", value="$__all"))
      .include_all(include_all=True)
      .multi(multi=True),
    )
    .with_variable(
      QueryVariable("interface")
      .label("interface")
      .query("label_values(smartctl_device, interface)")
      .current(VariableOption(selected=True, text="All", value="$__all"))
      .include_all(include_all=True)
      .multi(multi=True),
    )
    .with_variable(
      QueryVariable("model_name")
      .label("model_name")
      .query("label_values(smartctl_device, model_name)")
      .current(VariableOption(selected=True, text="All", value="$__all"))
      .include_all(include_all=True)
      .multi(multi=True),
    )
    .with_variable(
      QueryVariable("serial_number")
      .label("serial_number")
      .query("label_values(smartctl_device, serial_number)")
      .current(VariableOption(selected=True, text="All", value="$__all"))
      .include_all(include_all=True)
      .multi(multi=True),
    )
  )


def disk_temperature(builder: Dashboard) -> Dashboard:
  return builder.with_panel(
    Timeseries()
    .title("Disk Temperature")
    .unit("celsius")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_temperature{instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def raw_read_error_rate(builder: Dashboard) -> Dashboard:
  return builder.with_panel(
    Timeseries()
    .title("Raw Read Error Rate")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Raw_Read_Error_Rate", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def spin_up_time(builder: Dashboard) -> Dashboard:
  return builder.with_panel(
    Timeseries()
    .title("Spin Up Time")
    .unit("ms")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Spin_Up_Time", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def reallocated_sector_count(builder: Dashboard) -> Dashboard:
  return builder.with_panel(
    Timeseries()
    .title("Reallocated Sector Count")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Reallocated_Sector_Ct", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def start_stop_count(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Start/Stop Count")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Start_Stop_Count", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def seek_error_rate(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Seek Error Rate")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Seek_Error_Rate", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def power_on_hours(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Power On Hours")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Power_On_Hours", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def spin_retry_count(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Spin Retry Count")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Spin_Retry_Count", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def calibration_retry_count(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Calibration Retry Count")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Calibration_Retry_Count", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def power_cycle_count(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Power Cycle Count")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Power_Cycle_Count", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def power_off_retract_count(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Power Off Retract Count")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Power-Off_Retract_Count", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def load_cycle_count(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Load Cycle Count")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Load_Cycle_Count", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def reallocated_event_count(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Reallocated Event Count")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Reallocated_Event_Count", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def current_pending_sector_count(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Current Pending Sector Count")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Current_Pending_Sector", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def udma_crc_error_count(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("UDMA CRC Error Count")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="UDMA_CRC_Error_Count", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def offline_uncorrectable(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Offline Uncorrectable")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Offline_Uncorrectable", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def multi_zone_error_rate(builder: Row) -> Row:
  return builder.with_panel(
    Timeseries()
    .title("Multi-Zone Error Rate")
    .with_target(
      PrometheusQuery()
      .expr(
        """
        avg by (instance, device, model_name) (
          smartctl_device_attribute{attribute_name="Multi_Zone_Error_Rate", attribute_value_type="raw", instance=~'$node', device=~'$disk'} *
          on(instance, device) group_left(interface, serial_number, model_name)
          smartctl_device{interface=~'$interface', serial_number=~'$serial_number', model_name=~'$model_name'}
        )
        """,
      )
      .legend_format("{{ instance }} / {{ device }} / {{ model_name }}"),
    )
    .legend(
      VizLegendOptions()
      .display_mode(LegendDisplayMode.TABLE)
      .placement(LegendPlacement.BOTTOM)
      .calcs(["mean", "lastNotNull", "max", "min"])
      .show_legend(show_legend=True),
    ),
  )


def manifest() -> Manifest:
  dash = dashboard().build()

  return Manifest(
    api_version="dashboard.grafana.app/v1beta1",
    kind="Dashboard",
    metadata=Metadata(name=dash.uid or ""),
    spec=dash,
  )


if __name__ == "__main__":
  print(JSONEncoder(sort_keys=True, indent=2).encode(manifest()))
