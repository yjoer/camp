import { createORPCClient } from '@orpc/client';
import { RPCLink } from '@orpc/client/message-port';
import { MessageType } from '@orpc/standard-server-peer';
import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useEffect, useRef, useState } from 'react';

import { button_styles } from '@/components/button';
import CanvasWorker from '@/lib/canvas-worker?worker';

import type { router } from '@/lib/canvas-worker';
import type { RouterClient } from '@orpc/server';

export const Route = createFileRoute('/worker-offscreen-canvas')({
  component: OffscreenCanvas,
});

function OffscreenCanvas() {
  const ref = useRef<HTMLDivElement>(null!);
  const worker_ref = useRef<CanvasWorkerClient>(null);
  const [date, set_date] = useState<string>('');

  const handle_click = () => {
    if (!worker_ref.current) return;

    void worker_ref.current.get_date().then((date) => {
      set_date(date.toISOString());
    });
  };

  useEffect(() => {
    const canvas = document.createElement('canvas');
    ref.current.append(canvas);

    const offscreen = canvas.transferControlToOffscreen();
    const worker = get_worker_client();
    worker_ref.current = worker;

    transferables.add(offscreen);
    void worker.render({ canvas: offscreen });

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
          <button onClick={handle_click} {...stylex.props(button_styles.base)}>
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

const transferables = new WeakSet<Transferable>();

function get_worker_client(): CanvasWorkerClient {
  const link = new RPCLink({
    port: new CanvasWorker(),
    experimental_transfer: (message) => {
      const [_id, type, payload] = message;
      if (type !== MessageType.REQUEST) return [];

      const transfer: Transferable[] = [];
      const body = payload.body as { json: Record<string, unknown> };
      for (const v of Object.values(body.json ?? {})) {
        if (transferables.has(v as object)) transfer.push(v as Transferable);
      }

      return transfer;
    },
  });

  return createORPCClient(link);
}

type CanvasWorkerClient = RouterClient<typeof router>;
