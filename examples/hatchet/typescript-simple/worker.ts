import { hatchet } from './hatchet-client.ts';
import { hello } from './workflow.ts';

async function main() {
  const worker = await hatchet.worker('simple-worker', {
    workflows: [hello],
    slots: 100,
  });

  await worker.start();
}

await main();
