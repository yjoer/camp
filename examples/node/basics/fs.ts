// oxlint-disable no-console
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, open, readFile, unlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import * as asciichart from 'asciichart';

const fp = path.join(os.tmpdir(), 'camp', 'examples-node-basic', 'test.bin');

async function main() {
  const args = process.argv.slice(2);
  if (args.length === 0) return;

  switch (args[0]) {
    case 'bench': {
      const [u1] = await measure(usingReadFile);
      const [u2] = await measure(usingReadStream);
      const [u3] = await measure(usingSharedBuffer);

      const t1 = u1.map(u => u.arrayBuffers / 1_000_000);
      const t2 = u2.map(u => u.arrayBuffers / 1_000_000);
      const t3 = u3.map(u => u.arrayBuffers / 1_000_000);

      console.log(
        asciichart.plot([t1, t2, t3], {
          colors: [asciichart.lightblue, asciichart.lightgreen, asciichart.lightmagenta],
          height: 10,
        }),
      );

      break;
    }
    case 'clean': {
      await clean();
      break;
    }
    case 'prepare': {
      await prepare();
      break;
    }
    case 'read-file': {
      await usingReadFile();
      break;
    }
    case 'read-stream': {
      await usingReadStream();
      break;
    }
    case 'shared-buffer': {
      await usingSharedBuffer();
      break;
    }
  }
}

await main();

async function measure(callback: () => Promise<void>) {
  let gc = false;
  const usages: any[] = [];
  const usagesGC: any[] = [];

  const interval = setInterval(() => {
    if (gc) {
      usagesGC.push(process.memoryUsage());
    } else {
      usages.push(process.memoryUsage());
    }
  }, 50);

  await Promise.resolve(callback());

  await new Promise<void>((resolve) => {
    setTimeout(() => {
      if (!globalThis.gc) return;

      console.log('forcing garbage collection');
      globalThis.gc();
      gc = true;
    }, 1000);

    setTimeout(() => {
      resolve();
      clearInterval(interval);
    }, 3000);
  });

  return [usages, usagesGC];
}

async function prepare() {
  const buf = Buffer.alloc(1 * 1024 * 1024);
  const blocks = 512;

  await mkdir(path.dirname(fp), { recursive: true });
  const stream = createWriteStream(fp);

  let count = 0;

  while (count < blocks) {
    stream.write(buf);
    count++;
  }

  stream.end(() => {
    console.log(`successfully wrote ${blocks} blocks to ${fp}`);
  });
}

async function clean() {
  await unlink(fp);
}

async function usingReadFile() {
  const buf = await readFile(fp);
  console.log(`read ${buf.length} bytes`);
}

async function usingReadStream() {
  const stream = createReadStream(fp, { highWaterMark: 8 * 1024 * 1024 });

  let length = 0;
  for await (const chunk of stream) {
    length += chunk.length;
  }

  console.log(`read ${length} bytes using read stream`);
}

async function usingSharedBuffer() {
  const chunks = readChunks(fp);

  let length = 0;
  for await (const chunk of chunks) {
    length += chunk.length;
  }

  console.log(`read ${length} bytes using shared buffer`);
}

async function* readChunks(fp: string) {
  const fd = await open(fp, 'r');
  const stats = await fd.stat();

  const buf = Buffer.alloc(8 * 1024 * 1024);
  const blocks = Math.ceil(stats.size / buf.length);
  let end = stats.size % buf.length;

  for (let i = 0; i < blocks; i++) {
    await fd.read(buf);

    if (i == blocks - 1 && end > 0) {
      yield buf.subarray(0, end);
      continue;
    }

    yield buf;
  }

  await fd.close();
}
