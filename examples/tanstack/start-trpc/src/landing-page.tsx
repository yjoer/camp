import type { FallbackProps } from 'react-error-boundary';

import { useSuspenseQuery } from '@tanstack/react-query';
import { createFileRoute } from '@tanstack/react-router';
import cookies from 'js-cookie';
import { ErrorBoundary } from 'react-error-boundary';

import { useTRPC } from '@/lib/trpc-provider';

export const Route = createFileRoute('/')({
	component: LandingPage,
});

function LandingPage() {
	const trpc = useTRPC();
	const { data } = useSuspenseQuery(trpc.public.queryOptions());

	return (
		<div className="mx-2 my-1">
			<div className="flex gap-2">
				<button className="font-medium" onClick={handle_set_cookie}>
					Set Cookie
				</button>
				<button className="font-medium" onClick={handle_clear_cookie}>
					Clear Cookie
				</button>
			</div>
			<div className="mt-2 text-sm text-black/60">Public</div>
			<div className="flex bg-neutral-100 px-2 py-1 font-mono">{JSON.stringify(data)}</div>
			<div className="mt-2 text-sm text-black/60">Protected</div>
			<ErrorBoundary FallbackComponent={ProtectedError}>
				<Protected />
			</ErrorBoundary>
		</div>
	);
}

const handle_set_cookie = () => {
	cookies.set('session_token', '123', { expires: 365 });
};

const handle_clear_cookie = () => {
	cookies.remove('session_token');
};

function Protected() {
	const trpc = useTRPC();
	const { data } = useSuspenseQuery(trpc.protected.queryOptions());

	return <div className="flex bg-neutral-100 px-2 py-1 font-mono">{JSON.stringify(data)}</div>;
}

function ProtectedError({ error }: FallbackProps) {
	return <div className="flex bg-neutral-100 px-2 py-1 font-mono">{(error as Error).message}</div>;
}
