// oxlint-disable no-console
// oxlint-disable no-process-exit
import { Client, Connection } from '@temporalio/client';

import { getLocationFromIP } from './workflows.ts';

async function run() {
  const connection = await Connection.connect({ address: process.env.TEMPORAL_ADDRESS });
  const client = new Client({ connection });

  const handle = await client.workflow.start(getLocationFromIP, {
    taskQueue: 'ip-geolocation',
    args: ['Brian'],
    workflowId: 'get-location-from-ip',
  });

  console.log(await handle.result());
}

try {
  await run();
} catch (error) {
  console.error(error);
  process.exit(1);
}
