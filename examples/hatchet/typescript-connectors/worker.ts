import { dropbox } from './connectors/dropbox.ts';
import { hatchet } from './hatchet-client.ts';

async function main() {
  const worker = await hatchet.worker('connectors-worker', {
    workflows: [dropbox],
    slots: 100,
  });

  await worker.start();
}

await main();
