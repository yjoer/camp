// oxlint-disable no-console
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, open, readFile, unlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const fp = path.join(os.tmpdir(), 'camp', 'examples-node-basic', 'test.bin');

async function main() {
  const interval = setInterval(() => {
    console.log(process.memoryUsage());
  }, 1000);

  const args = process.argv.slice(2);
  if (args.length === 0) return;

  switch (args[0]) {
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

  setTimeout(() => {
    if (!globalThis.gc) return;

    console.log('forcing garbage collection');
    globalThis.gc();
  }, 1000);

  setTimeout(() => {
    clearInterval(interval);
  }, 3000);
}

await main();

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
