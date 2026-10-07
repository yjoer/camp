import * as common from '@grafana/grafana-foundation-sdk/common';
import {
	manifest as _manifest,
	autoGrid,
	DashboardBuilder,
	DatasourceVariableBuilder,
	PanelBuilder,
	QueryGroupBuilder,
	QueryVariableBuilder,
	row,
	rows,
	tab,
	tabs,
	TargetBuilder,
	ThresholdsConfigBuilder,
	TimeSettingsBuilder,
	TransformationBuilder,
} from '@grafana/grafana-foundation-sdk/dashboardv2';
import { QueryV2Builder as PrometheusQueryBuilder, PromQueryFormat } from '@grafana/grafana-foundation-sdk/prometheus';
import { VisualizationV2Builder as StatBuilder } from '@grafana/grafana-foundation-sdk/stat';
import { VisualizationV2Builder as TableBuilder } from '@grafana/grafana-foundation-sdk/table';
import { VisualizationV2Builder as TimeseriesBuilder } from '@grafana/grafana-foundation-sdk/timeseries';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

function dashboard(): DashboardBuilder {
	let builder = new DashboardBuilder('PgBouncer Exporter')
	.tags(['prometheus'])
	.timeSettings(
		new TimeSettingsBuilder()
		.timezone('browser')
		.from('now-6h')
		.to('now')
		.autoRefresh('5s'),
	);

	builder = variables(builder);

	return builder
	.element('stat-up', stat_up())
	.element('stat-max-client-connections', stat_max_client_connections())
	.element('stat-max-user-connections', stat_max_user_connections())
	.element('stat-databases', stat_databases())
	.element('stat-users', stat_users())
	.element('stat-pools', stat_pools())
	.element('stat-cached-dns-names', stat_cached_dns_names())
	.element('stat-cached-dns-zones', stat_cached_dns_zones())
	.element('used-clients', used_clients())
	.element('used-servers', used_servers())
	.element('client-connections', client_connections())
	.element('server-connections', server_connections())
	.element('client-active-connections', client_active_connections())
	.element('client-waiting-connections', client_waiting_connections())
	.element('client-maxwait-seconds', client_maxwait_seconds())
	.element('client-active-cancel-connections', client_active_cancel_connections())
	.element('client-waiting-cancel-connections', client_waiting_cancel_connections())
	.element('server-active-connections', server_active_connections())
	.element('server-idle-connections', server_idle_connections())
	.element('server-active-cancel-connections', server_active_cancel_connections())
	.element('server-being-canceled-connections', server_being_canceled_connections())
	.element('server-used-connections', server_used_connections())
	.element('server-testing-connections', server_testing_connections())
	.element('server-login-connections', server_login_connections())
	.element('settings', settings())
	.layout(
		rows()
		.row(
			row('Summary')
			.collapse(false)
			.layout(
				rows()
				.row(
					row('')
					.collapse(false)
					.hideHeader(true)
					.layout(
						autoGrid()
						.maxColumnCount(3)
						.columnWidthMode('narrow')
						.rowHeightMode('short')
						.withItem('stat-up')
						.withItem('stat-max-client-connections')
						.withItem('stat-max-user-connections'),
					),
				)
				.row(
					row('')
					.collapse(false)
					.hideHeader(true)
					.layout(
						autoGrid()
						.maxColumnCount(5)
						.columnWidthMode('narrow')
						.rowHeightMode('short')
						.withItem('stat-databases')
						.withItem('stat-users')
						.withItem('stat-pools')
						.withItem('stat-cached-dns-names')
						.withItem('stat-cached-dns-zones'),
					),
				)
				.row(
					row('')
					.collapse(false)
					.hideHeader(true)
					.layout(
						autoGrid()
						.maxColumnCount(2)
						.withItem('used-clients')
						.withItem('used-servers')
						.withItem('client-connections')
						.withItem('server-connections'),
					),
				),
			),
		)
		.row(
			row('Pools')
			.collapse(false)
			.layout(
				tabs()
				.tab(
					tab('Client').layout(
						autoGrid()
						.maxColumnCount(2)
						.rowHeightMode('tall')
						.withItem('client-active-connections')
						.withItem('client-waiting-connections')
						.withItem('client-maxwait-seconds')
						.withItem('client-active-cancel-connections')
						.withItem('client-waiting-cancel-connections'),
					),
				)
				.tab(
					tab('Server').layout(
						autoGrid()
						.maxColumnCount(2)
						.rowHeightMode('tall')
						.withItem('server-active-connections')
						.withItem('server-idle-connections')
						.withItem('server-active-cancel-connections')
						.withItem('server-being-canceled-connections')
						.withItem('server-used-connections')
						.withItem('server-testing-connections')
						.withItem('server-login-connections'),
					),
				)
				.tab(
					tab('Settings').layout(
						autoGrid()
						.maxColumnCount(2)
						.withItem('settings'),
					),
				),
			),
		),
	);
}

function variables(builder: DashboardBuilder): DashboardBuilder {
	return builder
	.variable(
		new DatasourceVariableBuilder('datasource')
		.label('datasource')
		.pluginId('prometheus'),
	)
	.variable(
		new QueryVariableBuilder('instance')
		.label('instance')
		.query(
			new PrometheusQueryVariableBuilder()
			.datasource({ name: '$datasource' })
			.query('label_values(pgbouncer_version_info, instance)'),
		)
		.current({ selected: true, text: 'All', value: '$__all' })
		.includeAll(true)
		.multi(false),
	);
}

function stat_up(): PanelBuilder {
	return new PanelBuilder()
	.title('Status')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_up{instance=~'$instance'}"),
			),
		),
	)
	.visualization(
		new StatBuilder()
		.colorMode(common.BigValueColorMode.Background)
		.graphMode(common.BigValueGraphMode.None)
		.thresholds(
			new ThresholdsConfigBuilder().steps([
				{ value: 0, color: 'red' },
				{ value: 1, color: 'green' },
			]),
		),
	);
}

function stat_max_client_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Max Client Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_config_max_client_connections{instance=~'$instance'}"),
			),
		),
	)
	.visualization(
		new StatBuilder().graphMode(common.BigValueGraphMode.None),
	);
}

function stat_max_user_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Max User Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_config_max_user_connections{instance=~'$instance'}"),
			),
		),
	)
	.visualization(
		new StatBuilder().graphMode(common.BigValueGraphMode.None),
	);
}

function stat_databases(): PanelBuilder {
	return new PanelBuilder()
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_databases{instance=~'$instance'}")
				.legendFormat('Databases'),
			),
		),
	)
	.visualization(
		new StatBuilder()
		.graphMode(common.BigValueGraphMode.Area)
		.textMode(common.BigValueTextMode.ValueAndName)
		.text(new common.VizTextDisplayOptionsBuilder().titleSize(14).valueSize(64)),
	);
}

function stat_users(): PanelBuilder {
	return new PanelBuilder()
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_users{instance=~'$instance'}")
				.legendFormat('Users'),
			),
		),
	)
	.visualization(
		new StatBuilder()
		.graphMode(common.BigValueGraphMode.Area)
		.textMode(common.BigValueTextMode.ValueAndName)
		.text(new common.VizTextDisplayOptionsBuilder().titleSize(14).valueSize(64)),
	);
}

function stat_pools(): PanelBuilder {
	return new PanelBuilder()
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools{instance=~'$instance'}")
				.legendFormat('Pools'),
			),
		),
	)
	.visualization(
		new StatBuilder()
		.graphMode(common.BigValueGraphMode.Area)
		.textMode(common.BigValueTextMode.ValueAndName)
		.text(new common.VizTextDisplayOptionsBuilder().titleSize(14).valueSize(64)),
	);
}

function stat_cached_dns_names(): PanelBuilder {
	return new PanelBuilder()
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_cached_dns_names{instance=~'$instance'}")
				.legendFormat('Cached DNS Names'),
			),
		),
	)
	.visualization(
		new StatBuilder()
		.graphMode(common.BigValueGraphMode.Area)
		.textMode(common.BigValueTextMode.ValueAndName)
		.text(new common.VizTextDisplayOptionsBuilder().titleSize(14).valueSize(64)),
	);
}

function stat_cached_dns_zones(): PanelBuilder {
	return new PanelBuilder()
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_cached_dns_zones{instance=~'$instance'}")
				.legendFormat('Cached DNS Zones'),
			),
		),
	)
	.visualization(
		new StatBuilder()
		.graphMode(common.BigValueGraphMode.Area)
		.textMode(common.BigValueTextMode.ValueAndName)
		.text(new common.VizTextDisplayOptionsBuilder().titleSize(14).valueSize(64)),
	);
}

function used_clients(): PanelBuilder {
	return new PanelBuilder()
	.title('Used/Free Clients')
	.id(id++)
	.data(
		new QueryGroupBuilder()
		.target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_used_clients{instance=~'$instance'}")
				.legendFormat('Used'),
			),
		)
		.target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_free_clients{instance=~'$instance'}")
				.legendFormat('Free'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder()
		.fillOpacity(50)
		.stacking(new common.StackingConfigBuilder().mode(common.StackingMode.Normal))
		.legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.List)
			.placement(common.LegendPlacement.Bottom)
			.showLegend(true),
		),
	);
}

function used_servers(): PanelBuilder {
	return new PanelBuilder()
	.title('Used/Free Servers')
	.id(id++)
	.data(
		new QueryGroupBuilder()
		.target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_used_servers{instance=~'$instance'}")
				.legendFormat('Used'),
			),
		)
		.target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_free_servers{instance=~'$instance'}")
				.legendFormat('Free'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder()
		.fillOpacity(50)
		.stacking(new common.StackingConfigBuilder().mode(common.StackingMode.Normal))
		.legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.List)
			.placement(common.LegendPlacement.Bottom)
			.showLegend(true),
		),
	);
}

function client_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Client Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("sum by (instance, database, user) (pgbouncer_client_connections{instance=~'$instance'})")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder()
		.fillOpacity(50)
		.stacking(new common.StackingConfigBuilder().mode(common.StackingMode.Normal)).legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function server_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Server Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_databases_current_connections{instance=~'$instance'}")
				.legendFormat('{{ database }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder()
		.fillOpacity(50)
		.stacking(new common.StackingConfigBuilder().mode(common.StackingMode.Normal)).legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function client_active_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Client Active Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_client_active_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function client_waiting_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Client Waiting Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_client_waiting_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function client_maxwait_seconds(): PanelBuilder {
	return new PanelBuilder()
	.title('Client Max Wait')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_client_maxwait_seconds{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().unit('s').legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function client_active_cancel_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Client Active Cancel Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_client_active_cancel_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function client_waiting_cancel_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Client Waiting Cancel Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_client_waiting_cancel_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function server_active_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Server Active Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_server_active_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function server_idle_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Server Idle Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_server_idle_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function server_active_cancel_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Server Active Cancel Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_server_active_cancel_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function server_being_canceled_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Server Being Canceled Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_server_being_canceled_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function server_used_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Server Used Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_server_used_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function server_testing_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Server Testing Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_server_testing_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function server_login_connections(): PanelBuilder {
	return new PanelBuilder()
	.title('Server Login Connections')
	.id(id++)
	.data(
		new QueryGroupBuilder().target(
			new TargetBuilder().query(
				new PrometheusQueryBuilder()
				.datasource({ name: '$datasource' })
				.expr("pgbouncer_pools_server_login_connections{instance=~'$instance'}")
				.legendFormat('{{ database }} / {{ user }}'),
			),
		),
	)
	.visualization(
		new TimeseriesBuilder().legend(
			new common.VizLegendOptionsBuilder()
			.displayMode(common.LegendDisplayMode.Table)
			.placement(common.LegendPlacement.Bottom)
			.calcs(['mean', 'lastNotNull', 'max', 'min'])
			.showLegend(true),
		),
	);
}

function settings(): PanelBuilder {
	// oxlint-disable-next-line unicorn/consistent-function-scoping
	const table_query = (ref_id: string, metric: string) => new TargetBuilder()
	.refId(ref_id)
	.query(
		new PrometheusQueryBuilder()
		.datasource({ name: '$datasource' })
		.expr(`pgbouncer_databases_${metric}{instance=~'$instance'}`)
		.format(PromQueryFormat.Table)
		.instant(true),
	);

	return new PanelBuilder()
	.id(id++)
	.data(
		new QueryGroupBuilder()
		.target(table_query('A', 'pool_size'))
		.target(table_query('B', 'reserve_pool'))
		.target(table_query('C', 'max_connections'))
		.target(table_query('D', 'paused'))
		.target(table_query('E', 'disabled'))
		.transformation(
			new TransformationBuilder()
			.group('filterFieldsByName')
			.options({
				include: { names: ['database', 'Value #A', 'Value #B', 'Value #C', 'Value #D', 'Value #E'] },
			}),
		)
		.transformation(
			new TransformationBuilder()
			.group('joinByField')
			.options({
				byField: 'database',
				mode: 'outerTabular',
			}),
		).transformation(
			new TransformationBuilder()
			.group('organize')
			.options({
				renameByName: {
					'Value #A': 'pool_size',
					'Value #B': 'reserve_pool_size',
					'Value #C': 'max_db_connections',
					'Value #D': 'paused',
					'Value #E': 'disabled',
				},
			}),
		),
	)
	.visualization(
		new TableBuilder().footer(
			new common.TableFooterOptionsBuilder().show(true).reducer(['sum']),
		),
	);
}

let id = 0;

export function manifest() {
	return _manifest('pgbouncer-exporter', dashboard());
}

class PrometheusQueryVariableBuilder extends PrometheusQueryBuilder {
	query(expr: string): this {
		if (!this.internal.spec) this.internal.spec = {};
		(this.internal.spec as { query: string }).query = expr;
		return this;
	}
}

if (path.resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url)) {
	process.stdout.write(JSON.stringify(manifest().build(), undefined, 2));
}
