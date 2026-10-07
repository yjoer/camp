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
	TimeSettingsBuilder,
	TransformationBuilder,
} from '@grafana/grafana-foundation-sdk/dashboardv2';
import { QueryV2Builder as PrometheusQueryBuilder, PromQueryFormat } from '@grafana/grafana-foundation-sdk/prometheus';
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
		rows().row(
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
