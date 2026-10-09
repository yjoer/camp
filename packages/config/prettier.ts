import type { Config } from 'prettier';

export const prettier_config = {
	plugins: [
		// '@prettier/plugin-oxc',
		'@prettier/plugin-xml',
		'prettier-plugin-packagejson',
		// 'prettier-plugin-tailwindcss',
		// 'prettier-plugin-classnames',
		// 'prettier-plugin-merge',
		'prettier-plugin-java',
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
