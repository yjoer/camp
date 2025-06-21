import { hello } from './workflow.ts';

const res = await hello.run({
  message: 'Hello, World!',
});

// oxlint-disable-next-line no-console
console.log(res.transformedMessage);
