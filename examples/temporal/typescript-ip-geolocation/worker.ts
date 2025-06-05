/* eslint-disable no-console */
/* eslint-disable unicorn/no-process-exit */
import { fileURLToPath } from 'node:url';

import { NativeConnection, Worker } from '@temporalio/worker';

import * as activities from './activities.ts';

async function run() {
  const connection = await NativeConnection.connect({
    address: process.env.TEMPORAL_ADDRESS,
  });

  try {
    const worker = await Worker.create({
      connection,
      namespace: 'default',
      taskQueue: 'ip-geolocation',
      activities,
      workflowsPath: fileURLToPath(import.meta.resolve('./workflows.ts')),
    });

    await worker.run();
  } finally {
    await connection.close();
  }
}

try {
  await run();
} catch (error) {
  console.error(error);
  process.exit(1);
}
