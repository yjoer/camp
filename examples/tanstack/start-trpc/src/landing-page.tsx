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

  const handleSetCookie = () => {
    cookies.set('session_token', '123', { expires: 365 });
  };

  const handleClearCookie = () => {
    cookies.remove('session_token');
  };

  return (
    <div className="mx-2 my-1">
      <div className="flex gap-2">
        <button className="font-medium" onClick={handleSetCookie}>
          Set Cookie
        </button>
        <button className="font-medium" onClick={handleClearCookie}>
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

function Protected() {
  const trpc = useTRPC();
  const { data } = useSuspenseQuery(trpc.protected.queryOptions());

  return <div className="flex bg-neutral-100 px-2 py-1 font-mono">{JSON.stringify(data)}</div>;
}

interface ProtectedErrorProps {
  error: Error;
}

function ProtectedError({ error }: ProtectedErrorProps) {
  return <div className="flex bg-neutral-100 px-2 py-1 font-mono">{error.message}</div>;
}
