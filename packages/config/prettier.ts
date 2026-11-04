import type { Config, Plugin } from 'prettier';

import xml_plugin from '@prettier/plugin-xml';
import java_plugin from 'prettier-plugin-java';
import packagejson_plugin from 'prettier-plugin-packagejson';

export const prettier_config = {
	plugins: [
		xml_plugin,
		packagejson_plugin,
		java_plugin as Plugin,
	],
	bracketSameLine: true,
	printWidth: 100,
	useTabs: true,
	quoteProps: 'as-needed',
	singleQuote: true,
	//
	xmlWhitespaceSensitivity: 'ignore',
	// tailwindFunctions: ['clsx', 'cva'],
	// customFunctions: ['clsx', 'cva'],
	// endingPosition: 'absolute',
	overrides: [{
		files: '*.jsonc',
		options: {
			printWidth: 120,
		},
	}, {
		files: 'pnpm-lock.yaml',
		options: {
			printWidth: 200,
			useTabs: false,
			singleQuote: true,
			bracketSpacing: false,
			parser: 'yaml',
		},
	}],
} satisfies Config;
