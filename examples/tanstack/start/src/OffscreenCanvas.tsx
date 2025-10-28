import { createORPCClient } from '@orpc/client';
import { RPCLink } from '@orpc/client/message-port';
import { MessageType } from '@orpc/standard-server-peer';
import { createFileRoute } from '@tanstack/react-router';
import { useEffect, useRef, useState } from 'react';

// oxlint-disable-next-line import/default
import CanvasWorker from '../lib/canvas-worker?worker';

import type { router } from '../lib/canvas-worker';
import type { RouterClient } from '@orpc/server';

export const Route = createFileRoute('/worker-offscreen-canvas')({
  component: OffscreenCanvas,
});

function OffscreenCanvas() {
  const ref = useRef<HTMLDivElement>(null!);
  const worker_ref = useRef<CanvasWorkerClient>(null);
  const [date, set_date] = useState<string>('');

  const handle_click = async () => {
    if (!worker_ref.current) return;

    const date = await worker_ref.current.get_date();
    set_date(date.toISOString());
  };

  useEffect(() => {
    const canvas = document.createElement('canvas');
    ref.current.append(canvas);

    const offscreen = canvas.transferControlToOffscreen();
    const worker = get_worker_client();
    worker_ref.current = worker;

    transferables.add(offscreen);
    worker.render({ canvas: offscreen });

    return () => {
      canvas.remove();
    };
  }, []);

  return (
    <div className="mx-2 my-1 flex flex-col gap-4">
      <div>
        <span className="bg-[#ffdd00]">Date</span>
        <div className="h-6">{date}</div>
        <div className="mt-1">
          <button
            className="cursor-pointer rounded bg-black/4 px-3 py-1 transition-transform
              active:scale-96"
            onClick={handle_click}>
            Get Date
          </button>
        </div>
      </div>
      <div ref={ref}>
        <span className="bg-[#ffdd00]">Canvas</span>
      </div>
    </div>
  );
}

const transferables = new WeakSet<any>();

function get_worker_client() {
  const link = new RPCLink({
    port: new CanvasWorker(),
    experimental_transfer: (message) => {
      const [_id, type, payload] = message;
      if (type !== MessageType.REQUEST) return [];

      const transfer: any[] = [];
      const body = payload.body as { json: Record<string, any> };
      for (const v of Object.values(body.json ?? {})) {
        if (transferables.has(v)) transfer.push(v);
      }

      return transfer;
    },
  });

  return createORPCClient(link) as RouterClient<typeof router>;
}

type CanvasWorkerClient = RouterClient<typeof router>;
