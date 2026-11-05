declare module 'postcss-styled-syntax' {
	import type { Syntax } from 'postcss';

	const syntax: Syntax;
	export default syntax;
}

declare module 'stylelint-config-recommended' {
	import type { Config } from 'stylelint';

	const config: Config;
	export default config;
}

declare module 'stylelint-config-standard' {
	import type { Config } from 'stylelint';

	const config: Config;
	export default config;
}

declare module 'stylelint-order' {
	import type { Plugin } from 'stylelint';

	const plugins: Plugin[];
	export default plugins;
}
